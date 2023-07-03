import argparse
import dgl
from dgl.dataloading import (
DataLoader,
    NeighborSampler,
    as_edge_prediction_sampler,
    negative_sampler,
)
import numpy as np
import pandas as pd
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from graphsage_model import Model

def evaluate(model, val_dataloader):
    model.eval()
    score_all = torch.Tensor()
    labels_all = torch.Tensor()
    with val_dataloader.enable_cpu_affinity():
        with torch.no_grad():
            for _, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                tqdm.tqdm(val_dataloader)
            ):
                x = input_nodes
                pos_score, pos_label = model(pair_graph, neg_pair_graph, blocks, x)
                score_all = torch.cat([score_all, pos_score], 0)
                labels_all = torch.cat([labels_all, pos_label], 0)
    assert(score_all.shape[0] == labels_all.shape[0])
    labels_one_hot = torch.hstack((1.0 - labels_all, labels_all)).float()
    bceloss = F.binary_cross_entropy(score_all, labels_one_hot)
    return bceloss


def train(args, train_dataloader, val_dataloader, model):
    best_model_path = args.model_out
    best_val_bceloss = float("inf")
    print("learning rate: ", args.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    with train_dataloader.enable_cpu_affinity():
        all_losses = []
        for epoch in range(args.num_epochs):
            start = time.time()
            step_time = []
            model.train()
            total_loss = []
            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                train_dataloader
            ):
                opt.zero_grad()
                x = input_nodes #blocks[0].srcdata[dgl.NID] #this is equal to input_nodes
                pos_score, pos_label = model(pair_graph, neg_pair_graph, blocks, x)
                pos_label_oh = torch.hstack((1.0 - pos_label, pos_label)).float() # 0 -> [1, 0] and 1 -> [0, 1]
                loss = F.binary_cross_entropy(pos_score, pos_label_oh)
                total_loss += loss.detach().item(),
                loss.backward()
                opt.step()
                
                step_t = time.time() - start
                step_time.append(step_t)
                start = time.time()

            all_losses += total_loss
            print("Epoch {:02d} | Train BCELoss {:.4f}".format(epoch, sum(total_loss) / len(train_dataloader)))

            model.eval()
            if (epoch % args.eval_every == 0) or (epoch == args.num_epochs -1):
                bceloss = evaluate(model, val_dataloader)
                # update best model if needed
                if best_val_bceloss > bceloss:
                    print("Updating best model")
                    best_val_bceloss = bceloss
                    torch.save(model.state_dict(), best_model_path)
                print("Epoch {:05d} | val bceloss {:.4f}".format(epoch, bceloss))

def main(arglist):
    # random seeds for testing
    seed=7
    print("random seed set to: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "gpu", "mixed"],
        help="Training mode. 'cpu' for CPU training, 'gpu' for pure-GPU training, "
        "'mixed' for CPU-GPU mixed training.",
    )
    parser.add_argument(
        "--CSVDataset_dir",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/recsys_graph",
        help="Path to CSVDataset",
    )
    parser.add_argument(
        "--model_out",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/gnn_model.pt",
        type=str,
        help="output for model /your_path/model.pt",
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_hidden_gnn", type=int, default=128)
    parser.add_argument("--num_hidden_mlp", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="25,25")
    parser.add_argument("--batch_size", type=int, default=40000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    # step 1 : parse input arguments
    args = parser.parse_args(arglist)
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # step 2 : load CSVDataset files into a heterogeneous graph dataset
    print("Loading data")
    start = time.time()
    # note : set force_reload=False if no changes on input graph (much faster otherwise ingestion ~30min)
    dataset = dgl.data.CSVDataset(args.CSVDataset_dir, force_reload=False)
    print("time to load dataset from CSVs: ", time.time() - start)

    # step 3 : extract the heterogenous graph
    train_val_hg = dataset[0]
    print(train_val_hg)
    print("etype to read train/test/val from: ", train_val_hg.canonical_etypes[0][1])
    print("-"*100)
    train_val_g = dgl.to_homogeneous(train_val_hg, 
                                     edata=["feat", "label", "train_mask", "val_mask", "test_mask"])
    # g.edata['prob'] = 1.0 - g.edata['test_mask']
    print(train_val_g)
    print("-"*100)

    # step 4 : create masks and collect edge IDs from 
    def get_mask_and_eids(graph, key):
        forward_mask = graph.edges["e"].data[key]
        reverse_mask = graph.edges["sym_e"].data[key]
        mask = torch.cat((forward_mask, reverse_mask))
        eids = torch.nonzero(mask, as_tuple=False).squeeze()
        return mask, eids
    
    full_val_mask, full_val_eids = get_mask_and_eids(train_val_hg, "val_mask")
    print("number of validation edges = {}".format(torch.sum(full_val_mask)))
    print("-"*100)

    # step 5 : create homogenous split graphs by removing edges val and/or test edges
    def remove_edges(graph, eids_to_remove):
        new_g = dgl.remove_edges(graph, eids_to_remove, store_ids=True)
        eidx_orig = new_g.edata[dgl.EID]
        assert len(torch.unique(eidx_orig)) == eidx_orig.shape[0] == new_g.num_edges()
        return new_g

    train_g = remove_edges(train_val_g, full_val_eids)
    num_train_edges = train_g.num_edges()
    num_val_edges = train_val_g.num_edges() - train_g.num_edges()
    print("train edges = {}, val edges = {}".format(num_train_edges, num_val_edges))
    print("-"*100)
    
    # step 6 : create sampler & dataloaders
    def get_reverse_mapping(graph):
        """
            pass homogeneous graph, 
            assumes forward edge with ID i has reverse edge with ID E + i
        """
        E = graph.num_edges() // 2
        reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])
        return reverse_eids
    
    sampler = NeighborSampler([int(fanout) for fanout in args.fan_out.split(",")])
    train_sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=get_reverse_mapping(train_g),
        negative_sampler=negative_sampler.Uniform(1),
    )
    val_sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=get_reverse_mapping(train_val_g),
        negative_sampler=negative_sampler.Uniform(1),
    )
    use_uva = args.mode == "mixed"

    # step 7 : collect edge IDs for forward edges in all three graphs
    def get_ids(graph, split):
        """
            assumption:
            -----------
                edge i has reverse edge (graph.num_edges() // 2 + i) and viceversa

            args:
            -----
                graph - homogenous graph with forward and backward edges 
                split - str among ["train_mask", "val_mask", "test_mask"]

            returns:
            --------
            returns forward edges' IDs for the defined split
        """
        mask = graph.edata[split] # mask will be of size graph.num_edges()
        num_edges = mask.shape[0]
        assert(num_edges % 2 == 0)
        eidx = torch.nonzero(mask[:num_edges//2], as_tuple=False).squeeze()
        return eidx
    
    # sanity check 1
    train_eidx = get_ids(train_g, "train_mask")
    num_train_edges = torch.sum(train_val_hg.edges[train_val_hg.canonical_etypes[0][1]].data["train_mask"]).int()
    assert(train_eidx.shape[0] == num_train_edges)

    # sanity check 2
    val_eidx = get_ids(train_val_g, "val_mask")
    num_val_edges = torch.sum(train_val_hg.edges[train_val_hg.canonical_etypes[0][1]].data["val_mask"]).int()
    assert(val_eidx.shape[0] == num_val_edges)

    train_dataloader = DataLoader(
        train_g,
        get_ids(train_g, "train_mask"),
        train_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        train_val_g,
        get_ids(train_val_g, "val_mask"),
        val_sampler,
        batch_size=args.batch_size_eval,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        use_uva=use_uva,
    )

    vocab_size = train_val_g.num_nodes()
    model = Model(vocab_size, args.num_hidden_gnn, args.num_hidden_gnn, args.num_hidden_mlp, args.num_layers)

    # model training
    print("Training...")
    train(args, train_dataloader, val_dataloader, model)




if __name__ == "__main__":
    main(sys.argv[1:])
