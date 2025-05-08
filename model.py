import torch 
import torch.nn as nn
from torch_geometric.nn import GATConv, GINConv, DeepGCNLayer, global_mean_pool, AttentionalAggregation, MFConv

class GAT(nn.Module):
    # captures connectivity importances
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 32)
        self.conv2 = GATConv(32, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class DeepGAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepGAT, self).__init__()
        hidden_channels = 32

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
    
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super(GIN, self).__init__()

        # Define a simple MLP used by GINConv
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv2 = GINConv(self.mlp2)

        # Final output projection (if needed)
        self.out_lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.out_lin(x)
        return x

class MFNet(nn.Module):
    # captures connectivity importances
    def __init__(self, in_channels, out_channels):
        super(MFNet, self).__init__()
        self.conv1 = MFConv(in_channels, 32)
        self.conv2 = MFConv(32, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_channels_cls, in_channels_mean, out_channels):
        super(MLP, self).__init__()
        self.cls_fc = nn.Sequential(
            nn.Linear(in_channels_cls, 128),
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.mean_fc = nn.Sequential(
            nn.Linear(in_channels_mean, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(64 + 64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
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

        hidden_dim = 128
        self.gat = GAT(in_channels, out_channels)
        self.deep_gat = DeepGAT(in_channels, out_channels)
        self.gin = GIN(in_channels, hidden_dim=hidden_dim, out_dim=out_channels)
        self.mlp = MLP(in_channels_cls, in_channels_mean, out_channels)
        self.mfnet = MFNet(in_channels, out_channels)

        # Attentional aggregator across branches
        self.branch_attention = AttentionalAggregation(
            gate_nn = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1)
            ),
            nn = nn.Identity()  # no additional transformation on input branches
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def forward(self, graph_data, cls_embed, mean_embed):
        # Graph data
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch

        mfnet_out = self.mfnet(x, edge_index)
        gat_out = self.gat(x, edge_index)
        deep_gat_out = self.deep_gat(x, edge_index)
        gin_out = self.gin(x, edge_index)

        # Pooling
        mfnet_pooled = global_mean_pool(mfnet_out, batch)
        gat_pooled = global_mean_pool(gat_out, batch)
        deep_gat_pooled = global_mean_pool(deep_gat_out, batch)
        gin_pooled = global_mean_pool(gin_out, batch)

        embedding_out = self.mlp(cls_embed, mean_embed)

        # Stack branch outputs shape [batch_size, num_branches, out_channels] for attention
        stacked = torch.stack([
            mfnet_pooled,
            gat_pooled,
            deep_gat_pooled,
            gin_pooled,
            embedding_out
        ], dim=1)

        # AttentionalAggregation
        out = self.branch_attention(stacked)

        # Final prediction head
        out = self.final_mlp(out)

        return out
    
if __name__ == "__main__":
    # Example usage
    model = twoTrackNetwork(in_channels=10, out_channels=1, in_channels_cls=600, in_channels_mean=600)
    print(model)