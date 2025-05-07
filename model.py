import torch 
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GATConv, MessagePassing, DeepGCNLayer, global_mean_pool

class GAT(nn.Module):
    # captures connectivity importances
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 16)
        self.conv2 = GATConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class DeepGAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepGAT, self).__init__()
        hidden_channels = 16

        # First layer
        self.input_conv = GATConv(in_channels, hidden_channels)

        # Deep GCN layer
        self.layer = DeepGCNLayer(
            conv=GATConv(hidden_channels, hidden_channels),
            norm=nn.BatchNorm1d(hidden_channels),
            act=torch.relu,
            block='res+',   # residual block
            dropout=0.1
        )

        # Output layer
        self.output_conv = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.input_conv(x, edge_index)
        x = self.layer(x, edge_index)
        x = self.output_conv(x, edge_index)
        return x
    
    
class MPNN(MessagePassing):
    # Captures richer edge and node interactions
    def __init__(self, in_channels, out_channels):
        super(MPNN, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):

        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return self.lin(x_j)

    def update(self, aggr_out):
        return self.lin2(aggr_out)
    
class MLP(nn.Module):
    def __init__(self, in_channels_cls, in_channels_mean, out_channels):
        super(MLP, self).__init__()
        self.cls_fc = nn.Sequential(
            nn.Linear(in_channels_cls, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.mean_fc = nn.Sequential(
            nn.Linear(in_channels_mean, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(64 + 64, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels)
        )

    def forward(self, cls_input, mean_input):
        cls_out = self.cls_fc(cls_input)
        mean_out = self.mean_fc(mean_input)
        combined = torch.cat([cls_out, mean_out], dim=1)
        out = self.final_fc(combined)
        return out

class twoTrackNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, in_channels_cls, in_channels_mean):
        super(twoTrackNetwork, self).__init__()
        self.gat = GAT(in_channels, out_channels)
        self.deep_gat = DeepGAT(in_channels, out_channels)
        self.mpnn = MPNN(in_channels, out_channels)
        self.mlp = MLP(in_channels_cls, in_channels_mean, out_channels)

        self.agg1 = nn.Linear(out_channels * 4, out_channels)
        self.agg2 = nn.Linear(out_channels, 1)

    def forward(self, graph_data, cls_embed, mean_embed):
        # Graph data
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        gat_out = self.gat(x, edge_index)
        deep_gat_out = self.deep_gat(x, edge_index)
        mpnn_out = self.mpnn(x, edge_index)

        # graph level data pooled
        gat_out_pooled = global_mean_pool(gat_out, batch)
        deep_gat_out_pooled = global_mean_pool(deep_gat_out, batch)
        mpnn_out_pooled = global_mean_pool(mpnn_out, batch)

        # Embedding data
        embedding_out = self.mlp(cls_embed, mean_embed)

        # Concatenate all outputs
        out = torch.cat((gat_out_pooled, deep_gat_out_pooled, mpnn_out_pooled, embedding_out), dim=1)
        
        # Final MLP
        out = torch.relu(self.agg1(out))
        out = self.agg2(out)
        
        return out
    
if __name__ == "__main__":
    # Example usage
    model = twoTrackNetwork(in_channels=10, out_channels=1, in_channels_cls=600, in_channels_mean=600)
    print(model)