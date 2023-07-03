from dgl.nn import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, n_layers, activation, aggregator_type, p=0.5
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(
                SAGEConv(in_feats, out_feats, aggregator_type)
                         #feat_drop=p, norm=nn.BatchNorm1d(out_feats), activation=activation)
                )
        else:
            # input layer
            self.layers.append(
                SAGEConv(in_feats, hidden_size, aggregator_type) 
                         #feat_drop=p, norm=nn.BatchNorm1d(hidden_size), activation=activation)
                )
            # hidden layers
            for i in range(1, n_layers - 1):
                self.layers.append(
                    SAGEConv(hidden_size, hidden_size, aggregator_type) 
                             #feat_drop=p, norm=nn.BatchNorm1d(hidden_size), activation=activation)
                    )
            # output layer
            self.layers.append(
                    SAGEConv(hidden_size, out_feats, aggregator_type) 
                             #feat_drop=p, norm=nn.BatchNorm1d(out_feats), activation=activation)
                         )

    def forward(self, graphs, inputs):
        h = inputs
        for (layer, block) in zip(self.layers, graphs):
            h = layer(block, h)
        return h 
    

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            # first layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),  
            # second layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            # third layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            # fourth layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            # fifth layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            # output layer
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h 


class Model(nn.Module):
    def __init__(self, vocab_size, gnn_hid_size, in_feat, mlp_hid_size, n_layers, p=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.gnn_hid_size = gnn_hid_size
        self.mlp_hid_size = mlp_hid_size
        # node embedding
        self.emb = torch.nn.Embedding(vocab_size, gnn_hid_size)
        # encoder is a 2-layer GraphSAGE model
        self.encoder = GraphSAGE(in_feat, gnn_hid_size, gnn_hid_size, n_layers, F.relu, "mean")
        # decoder is a 3-layer MLP
        self.decoder = Decoder(input_size=2*gnn_hid_size + 75, hidden_size=mlp_hid_size)
        self.batch_norm = nn.BatchNorm1d(2*gnn_hid_size + 75)
        self.dropout = nn.Dropout(p=p)
        self.act = nn.Softmax

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        """only for train and val. not for test"""
        # step 1 : node Ids -> emb -> hidden states for nodes
        h = self.emb(x)
        h = self.encoder(blocks, h) # size (num_nodes_in_batch, hidden_dim)
        # step 2 : get hidden states for the batch and concat edge features to them
        pos_src, pos_dst = pair_graph.edges()
        augmented_feat = torch.hstack((h[pos_src], h[pos_dst], pair_graph.edata['feat']))
        augmented_feat = self.batch_norm(augmented_feat)
        if self.training is True:
            augmented_feat = self.dropout(augmented_feat)
        # step 3 : run decoder on augmented features 
        h_pos = self.decoder(augmented_feat) # (bs, 2)
        # step 4 : get predictions and labels of size (bs, 2)
        h_pos = F.softmax(h_pos, dim=1) # (bs, 2)
        labels = pair_graph.edata['label'].reshape((-1, 1))  
        assert(labels.shape[0] == h_pos.shape[0])
        return h_pos, labels
    
    def predict(self, pair_graph, neg_pair_graph, blocks, x):
        """only for test, not for train and val"""
        self.eval()
        # step 1 : node Ids -> emb -> hidden states for nodes
        h = self.emb(x)
        h = self.encoder(blocks, h) # size (num_nodes_in_batch, hidden_dim)
        # step 2 : get hidden states for the batch and concat edge features to them
        pos_src, pos_dst = pair_graph.edges()
        augmented_feat = torch.hstack((h[pos_src], h[pos_dst], pair_graph.edata['feat']))
        augmented_feat = self.batch_norm(augmented_feat)
        # step 3 : run decoder on augmented features 
        h_pos = self.decoder(augmented_feat) # size (num_edges_in_batch, 2)
        # step 4 : get predictions and labels of size (num_edges_in_batch, 2)
        h_pos = F.softmax(h_pos, dim=1) # size (num_edges_in_batch, 2)
        return h_pos, pair_graph.edata['feat'][:, 0]
        