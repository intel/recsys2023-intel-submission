import argparse
import dgl
from dgl.dataloading import DataLoader, NeighborSampler
import numpy as np
import random
import sys
import time
import torch
import torch.nn.functional as F
import tqdm

sys.path.append('/localdisk/akakne/recsys2023/0_train/2_supGNN')
from graphsage_model import Model

def inference(model, g, batch_size):
        # step 1 : grab embeddings from model
        feat = model.emb.weight.data
        # IMPORTANT NOTE : we have set probability of selection of test edges to 0. (in line 112)
        # IMPORTANT NOTE : the line below ensures that - the model will only see one test edge at any given time. 
        sampler = NeighborSampler(fanouts=[-1], prob='prob')
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()), sampler,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=4)
        # step 3 : compure representations layer by layer, handle unseen nodes
        for l, layer in enumerate(model.encoder.layers):
            y = torch.empty(g.num_nodes(), model.gnn_hid_size)
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                    h = []
                    for node in input_nodes:
                        # If the node belongs to training data, get it's emb from the trained GNN model
                        if node < model.vocab_size:
                            h_node = feat[node]
                        # Else, it's an unseen node. Thus, will have a random embedding. 
                        else:
                            h_node = torch.rand(model.gnn_hid_size)
                        h += h_node,
                    h = torch.stack(h, dim=0)
                    h = layer(blocks[0], h)
                    if l != len(model.encoder.layers) - 1:
                        h = F.relu(h)
                    y[output_nodes] = h
                feat = y
        return y
        

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
        "--full_graph_CSVDataset_dir",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/full_graph/recsys_graph",
        help="Path to CSVDataset",
    )
    parser.add_argument(
        "--train_graph_CSVDataset_dir",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/recsys_graph",
        help="Path to CSVDataset",
    )
    parser.add_argument(
        "--model_path",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/train_graph/gnn_model.pt",
        type=str,
        help="output for model /your_path/model.pt",
    )
    parser.add_argument(
        "--nemb_out",
        default="/localdisk/akakne/recsys2023/data/supGNN_graph_data/full_graph/full_node_emb.pt",
        type=str,
        help="node emb output: /your_path/node_emb.pt",
    )
    # we need following arguments to initialize the GNN model prior to loading saved weights
    parser.add_argument("--num_hidden_gnn", type=int, default=128)
    parser.add_argument("--num_hidden_mlp", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size_eval", type=int, default=100000)

    # step 1 : parse input arguments
    args = parser.parse_args(arglist)
    if not torch.cuda.is_available():
        args.mode = "cpu"

    # step 2 : load CSVDataset files into a heterogeneous graph dataset
    print("Loading data")
    start = time.time()
    full_dataset = dgl.data.CSVDataset(args.full_graph_CSVDataset_dir, force_reload=False)
    print("time to load dataset from CSVs: ", time.time() - start)

    # step 3 : extract the heterogenous graph
    full_hg = full_dataset[0]
    print(full_hg)
    print("etype to read train/test/val from: ", full_hg.canonical_etypes[0][1])
    print("-"*100)
    full_g = dgl.to_homogeneous(full_hg, edata=["feat", "label", "train_mask", "val_mask", "test_mask"])
    # IMPORTANT NOTE : following line sets probablity of sampling of test edges to 0
    full_g.edata['prob'] = 1.0 - full_g.edata['test_mask']
    print(full_g)
    print("-"*100)

    # step 4 : load the saved GNN model and run inference on full graph
    def get_original_vocab_size(graph_path):
        hg = dgl.data.CSVDataset(graph_path, force_reload=False)[0]
        g = dgl.to_homogeneous(hg, edata=["feat", "label", "train_mask", "val_mask", "test_mask"])
        return g.num_nodes()
    
    vocab_size = get_original_vocab_size(args.train_graph_CSVDataset_dir)
    model = Model(vocab_size, args.num_hidden_gnn, args.num_hidden_gnn, args.num_hidden_mlp, args.num_layers)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    node_emb = inference(model, full_g, args.batch_size_eval)
    print("node emb shape: ", node_emb.shape)
    torch.save(node_emb, args.nemb_out)

if __name__ == "__main__":
    main(sys.argv[1:])