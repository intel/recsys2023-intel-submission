import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
import time
import torchmetrics.classification as MC


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.batch_norm = torch.nn.BatchNorm1d(hid_size)
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.decoder = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.batch_norm(h)
                h = F.relu(h)
                h = self.dropout(h)
        #h=F.softmax(h)
        h=self.decoder(h)
        return h

def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["is_installed"])
            y_hats.append(model(blocks, x))
    bceloss = F.binary_cross_entropy_with_logits(torch.cat(y_hats).flatten(), torch.cat(ys).float())
    return bceloss


def get_sample_weights(labels):
    class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    sample_weights = torch.tensor([weight[t] for t in labels])
    return sample_weights

def train(args, device, g, dataset, model, num_classes):
    
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)

    sampler_prob = NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")],
        prefetch_node_feats=["feat"],
        prefetch_labels=["is_installed"],
        prob='prob',
    )

    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler_prob,
        device=device,
        batch_size=1024, #2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler_prob,
        device=device,
        batch_size=1024, #2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_bceloss = float("inf")
    for epoch in range(int(args.num_epochs)):
        start=time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["is_installed"]
            y_hat = model(blocks, x)
            loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        epoch_time = time.time() - start
        bceloss= evaluate(model, g, val_dataloader, num_classes)
        print(
            "Epoch {:05d} | Epoch Time {:.4f} |Loss {:.4f} | valid BCELoss {:.4f} ".format(
                epoch,epoch_time, total_loss / (it + 1), bceloss,
            )
        )
        
        # update best model if needed
        if best_bceloss > bceloss:
            print("Updating best model")
            best_bceloss = bceloss
            torch.save(model.state_dict(), args.model_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--model_out",
        default="./2L_SAGE_45-66_graph7.pt",
        help="Filename for output model ",
    )
    parser.add_argument(
        "--nemb_out",
        #default="/GNN_TMP/sym_tabformer_hetero_CSVDataset_FINAL/node_emb.pt",
        default="./node_emb_graph7.pt",
        help="Filename for output embedding ",
    )
    parser.add_argument(
        "--num_epochs",
        default=10,
    )
    parser.add_argument("--fan_out", type=str, default="5,5,5")
    parser.add_argument(
        "--train_graph",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = dgl.data.CSVDataset(args.train_graph, force_reload=False)
    g = dataset[0]
    g.apply_edges(lambda edges: {'prob' : torch.abs(edges.dst['is_installed']-0.2) })
    print(g)
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = 1
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    hid_size=256
    out_size = hid_size
    model = SAGE(in_size, hid_size, out_size).to(device)

    # model training
    print("Training...")
    train(args, device, g, dataset, model, num_classes)

