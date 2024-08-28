import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, GCNConv, GATConv, GCNConv

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = SplineConv(dataset.num_features, 32, dim=2, kernel_size=4, is_open_spline=True, degree=3, aggr="max", root_weight=True, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=4, is_open_spline=True, degree=3, aggr="max", root_weight=True, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.conv3 = SplineConv(64, 128, dim=2, kernel_size=4, is_open_spline=True, degree=3, aggr="max", root_weight=True, bias=True)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.conv4 = SplineConv(128, 1, dim=2, kernel_size=4, is_open_spline=True, degree=3, aggr="max", root_weight=True, bias=True)
        self.bn4 = torch.nn.BatchNorm1d(64)
        # self.conv5 = SplineConv(64, 1, dim=2, kernel_size=4, is_open_spline=True, degree=2, aggr="add", root_weight=True, bias=True)
        # self.bn5 = torch.nn.BatchNorm1d(1)
        self.fc = torch.nn.Linear(1,1)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.elu(x)
        # x = self.fc(x)  # Linear layer for adjusting outputs before final activation
        x = torch.sigmoid(x)  # Sigmoid activation to output probabilities
        return x

class SplineCNN_SuperPixelNet(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # First SplineConv layer
        self.conv1 = SplineConv(dataset.num_features, 32, dim=2, kernel_size=4, is_open_spline=True, degree=2, aggr="add", root_weight=True, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.pool1 = torch.nn.MaxPool2d(4)  # Pooling layers might not be directly applicable depending on your graph structure

        # Second SplineConv layer
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=4, is_open_spline=True, degree=2, aggr="add", root_weight=True, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.pool2 = torch.nn.MaxPool2d(4)

        # Global Average Pooling
        self.glob_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Adjusted to graph's pooled feature dimensions

        # Fully connected layers
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # First SplineConv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        # Note: MaxPool2d is not directly applicable to graph data without a spatial feature representation

        # Second SplineConv layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        # Note: Again, MaxPool2d is not directly applicable here

        # Adjust pooling here to match graph data requirements
        # x = self.glob_avg_pool(x)  # Needs to be adapted if x is not a batched feature map

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        return x

class GCNNet(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = GCNConv(32, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.conv3 = GCNConv(64, 128)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.conv4 = GCNConv(128, 1)
        # Define more layers as needed, using torch.nn for any necessary components.

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv4(x, edge_index)
        x = F.elu(x)
        
        x = torch.sigmoid(x)  # Use sigmoid for binary classification problems
        return x

class GATNet(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, 32, heads=8, dropout=0.6)
        self.conv2 = GATConv(32 * 8, 64, heads=8, dropout=0.6)  # Adjust according to your model setup
        self.conv3 = GATConv(64 * 8, 128, heads=8, dropout=0.6)
        self.conv4 = GATConv(128 * 8, 1, heads=1, concat=False, dropout=0.6)  # Ensuring single output per node
        self.fc = torch.nn.Linear(1, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv4(x, edge_index)
        # x = self.fc(x) 
        x = torch.sigmoid(x)  # or no activation if purely regression without bounded output
        return x

class HybridGAT_SplineNet(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # Initialize GATConv with attention mechanism
        self.gat1 = GATConv(dataset.num_features, 32, heads=8, dropout=0.6)

        # Initialize SplineConv for capturing complex patterns with spline kernels
        self.spline1 = SplineConv(32 * 8, 64, dim=2, kernel_size=5, aggr='max')

        # Another layer of GATConv to focus on attention-based feature refinement
        self.gat2 = GATConv(64, 64, heads=8, dropout=0.6, concat=True)

        # Final SplineConv layer for detailed feature extraction before output
        self.spline2 = SplineConv(64 * 8, 1, dim=2, kernel_size=5, aggr='max')

        self.dropout_rate = 0.6

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        # Ensure pseudo coordinates are provided for SplineConv layers

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.spline1(x, edge_index, pseudo)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.spline2(x, edge_index, pseudo)

        x = torch.sigmoid(x)  # Assuming binary classification or sigmoid needed for final output

        return x

class ResidualGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        if in_channels != out_channels:
            self.projection = torch.nn.Linear(in_channels, out_channels)
        else:
            self.projection = None

    def forward(self, x, edge_index):
        identity = x
        out = F.relu(self.conv1(x, edge_index))
        out = self.conv2(out, edge_index)
        
        if self.projection is not None:
            identity = self.projection(identity)
        
        out += identity  # Residual connection
        out = F.relu(out)
        return out

class ResidualGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.initial_conv = GCNConv(in_channels, hidden_channels)
        self.res_blocks = torch.nn.ModuleList([ResidualGCNBlock(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
        self.final_conv = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.initial_conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for block in self.res_blocks:
            x = block(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.final_conv(x, edge_index)
        return x