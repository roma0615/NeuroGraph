import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, ModuleList
from torch_geometric.data import Data

from mace_layer import MACE_layer


# Multi-body potential (MBP) GNN layer
class MBPGNN(Module):
    # TODO CHANGE INPUTS
    # def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
    # def __init__(self,args, train_dataset, hidden_channels,hidden, num_layers, GNN, k=0.6):
    # model = MBPGNN(train_dataset, args.hidden, args.num_layers)
    def __init__(self, dataset, num_hidden, hidden_mlp, num_layers):
        """GNN that uses layer from MACE as message passing layer

        Args:
            num_features: (int) - number of features, AKA in dimension
            hidden_channels: (int) - hidden channels, AKA out dimension
        """
        super().__init__()
        self.convs = ModuleList()

        self.num_attributes = dataset.num_features # "features" are actually attributes; true node features are learned and change over time
        self.hidden_channels = num_hidden
        
        # Linear projection for initial node features
        # dim: d_n -> d
        # embedding dimension
        self.attrib_to_features = Linear(self.num_attributes, self.num_attributes) # get node "features" from attributes, to then pass through the mace layer

        for i in range(0, num_layers):
            # mace layer produces node features of same size, so we've already downscaled at this point
            self.convs.append(MACE_layer(
                max_ell=3,
                correlation=3,
                n_dims_in=self.num_attributes,
                hidden_irreps="256x0e + 256x1o", # recommended hidden model size (MACE repo)
                node_feats_irreps=f"{self.num_attributes}x0e",
                edge_feats_irreps="1x0e", # TODO what does this do?
                avg_num_neighbors=10.0,
                use_sc=True,
            ))

        input_dim1 = int(((self.num_attributes * self.num_attributes)/2)- (self.num_attributes/2)+(self.hidden_channels*num_layers))
        input_dim = int(((self.num_attributes * self.num_attributes)/2)- (self.num_attributes/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(self.hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_mlp, hidden_mlp//2),
            nn.BatchNorm1d(hidden_mlp//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_mlp//2, hidden_mlp//2),
            nn.BatchNorm1d(hidden_mlp//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden_mlp//2), dataset.num_classes),
        )
        self.softmax = torch.nn.LogSoftmax(dim=1)


    def forward(self, data: Data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """

        # TODO:
        # [ ] figure out what to pass in as vectors. its used to make spherical harmonics so it def matters.
        #   - vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3] -- from mace/modules/utils get_edge_vectors_and_lengths()
        # [X] figure out why edge_attr is wrong OHHH ITS A BATCH. but still why it expecting 64 :( oh thats a layer. oh im passing wrong shape of these things.
        #   - ok fixed the shape of stuff
        # [ ] can't concatenate tensors of diff lengths - diff elemenets in batch have diff # edges :(

        # from ResidualGNNs
        x = data.x
        xs = [x] # keep a running list of node features
        # right now, x is attribute
        xs += [self.attrib_to_features(xs[-1]).relu()]
        # now its a feature
        for mace_layer in self.convs:
            '''
            node_feats = self.mace_layer1(
                data.vectors, # vectors ??? what are these
                node_feats, # node_feats,
                data.x, # node_attrs,
                data.edge_attr, # edge_feats, THIS IS NONE what do we dooooo
                data.edge_index, # edge_index
            )
            '''
            xs += [mace_layer(
                data.edge_vectors.t(), # vectors (?)
                xs[-1], # node feats
                data.x, # node attributes
                data.edge_attr, # edge attr/feats (?)
                data.edge_index
            ).tanh()] 

        # what is this part. ig don't have to understand it
        h = []
        for i, xx in enumerate(xs): # for all values that node feats was
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, data.batch)
                h.append(xx)
        
        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        x = torch.cat((x,h),dim=1)
        x = self.mlp(x)
        return self.softmax(x)






        # use things such as
        # position encodings
        # angles????
        # node wise NLP followd by global readout
        # how do i do that.
        # do positions matter in brain.

        node_feats = self.node_embedding(data.x)
        # can i just give it constants as edge feats

        node_feats = self.mace_layer1(
            data.vectors, # vectors ??? what are these
            node_feats, # node_feats,
            data.x, # node_attrs,
            data.edge_attr, # edge_feats, THIS IS NONE what do we dooooo
            data.edge_index, # edge_index
        )
        return node_feats

        # out = self.mace_layer(
            # vectors, # ???
            # node_feats, # node feats. 
            # data.x, # node attrs (the intrinisc ones)
            # data.edge_attr, # 
            # data.edge_index
        # )
        
        # data.x = node feature matrix
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)

        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)