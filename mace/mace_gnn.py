import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import aggr
from e3nn import o3

from mace_layer import MACE_layer

class MaceGNN(Module):
    def __init__(self, dataset, num_hidden, hidden_mlp, num_layers):
        """GNN that uses layer from MACE as message passing layer

        Args:
            num_features: (int) - number of features, AKA in dimension
            hidden_channels: (int) - hidden channels, AKA out dimension
        """
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()

        self.num_attributes = dataset.num_features
        self.hidden_channels = num_hidden
        self.hidden_irreps = o3.Irreps("64x0e + 64x1o")
        self.hidden_irreps_out = self.hidden_irreps[0] # only scalars for last layer

        node_attr_irreps = o3.Irreps([(self.num_attributes, (0, 1))])
        node_feats_irreps = o3.Irreps([(self.hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = o3.Linear(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        self.convs.append(MACE_layer(
            max_ell=3,
            correlation=3,
            n_dims_in=self.num_attributes,
            node_feats_irreps=str(node_feats_irreps), 
            hidden_irreps=str(self.hidden_irreps), # recommended hidden model size (MACE repo)
            edge_feats_irreps="1x0e",
            avg_num_neighbors=10.0,
            use_sc=True,
        ))
        for i in range(num_layers-1):
            if i < num_layers - 2: 
                self.convs.append(MACE_layer(
                    max_ell=3,
                    correlation=3,
                    n_dims_in=self.num_attributes,
                    node_feats_irreps=str(self.hidden_irreps),
                    hidden_irreps=str(self.hidden_irreps),
                    edge_feats_irreps="1x0e",
                    avg_num_neighbors=10.0,
                    use_sc=True,
                ))
            else: # last layer
                self.convs.append(MACE_layer(
                    max_ell=3,
                    correlation=3,
                    n_dims_in=self.num_attributes,
                    node_feats_irreps=str(self.hidden_irreps),
                    hidden_irreps=str(self.hidden_irreps_out),
                    edge_feats_irreps="1x0e",
                    avg_num_neighbors=10.0,
                    use_sc=True,
                ))


        input_dim = int(((self.num_attributes * self.num_attributes)/2)- (self.num_attributes/2))
        self.bn = nn.BatchNorm1d(input_dim)
        bnh_input_dim = node_feats_irreps.dim + self.hidden_irreps.dim * (num_layers-1) + self.hidden_irreps_out.dim
        self.bnh = nn.BatchNorm1d(bnh_input_dim)

        input_dim1 = int(bnh_input_dim + (self.num_attributes**2 - self.num_attributes)//2)
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

        x = data.x
        # from random
        edge_vectors = data.edge_vectors.t()
        xs = [x]
        xs += [self.node_embedding(xs[-1]).tanh()]
        for mace_layer in self.convs:
            xs += [mace_layer(
                edge_vectors, # vectors
                xs[-1], # node feats
                data.x, # node attributes
                data.edge_attr, # edge attr/feats
                data.edge_index
            ).tanh()] 

        h = []
        for i, xx in enumerate(xs): 
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1) 
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx]) # get values just in top triangle (since its symmetric). this is n^2 - n values
                x = self.bn(x)
            else:
                xx = self.aggr(xx, data.batch)
                h.append(xx)
        
        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        x = torch.cat((x,h),dim=1)
        x = self.mlp(x)
        return self.softmax(x)
