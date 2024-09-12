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

class ConvolutionalAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = torch.nn.Conv2d(3, 84, kernel_size=3, stride=1, padding=1)  # Assuming no stride change
        self.pool1 = torch.nn.MaxPool2d(2, 2)  # MaxPooling
        self.conv2 = torch.nn.Conv2d(84, 168, kernel_size=3, stride=1, padding=1)  # Assuming no stride change
        self.pool2 = torch.nn.MaxPool2d(2, 2)  # MaxPooling
        self.conv3 = torch.nn.Conv2d(168, 336, kernel_size=3, stride=1, padding=1)  # Assuming no stride change
        self.pool3 = torch.nn.MaxPool2d(2, 2)  # MaxPooling
        # self.conv4 = torch.nn.Conv2d(336, 672, kernel_size=3, stride=1, padding=1)  # Assuming no stride change
        # self.pool4 = torch.nn.MaxPool2d(2, 2)  # MaxPooling

        # Decoder
        # self.t_conv1 = torch.nn.ConvTranspose2d(672, 336, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = torch.nn.ConvTranspose2d(336, 168, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = torch.nn.ConvTranspose2d(168, 84, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv4 = torch.nn.ConvTranspose2d(84, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoding path
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        print(f'After conv1 and pool: {x.size()}')
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        print(f'After conv2 and pool: {x.size()}')
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        print(f'After conv3 and pool: {x.size()}')
        # x = F.relu(self.conv4(x))
        # x = self.pool4(x)
        # print(f'After conv4 and pool: {x.size()}')

        # Decoding path
        # x = F.relu(self.t_conv1(x))
        print(f'After t_conv1: {x.size()}')
        x = F.relu(self.t_conv2(x))
        print(f'After t_conv2: {x.size()}')
        x = F.relu(self.t_conv3(x))
        print(f'After t_conv3: {x.size()}')
        x = torch.sigmoid(self.t_conv4(x))
        print(f'Final output: {x.size()}')
        return x
    
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask].unsqueeze(1))  # Use MSE loss for continuous output
#     loss.backward()
    
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()

# @torch.no_grad()
# def test():
#     model.eval()
#     out = model(data)  # Output shape: [num_nodes, 1]
#     out = out.squeeze()  # Squeeze to match target shape [num_nodes]

#     train_error = F.mse_loss(out[data.train_mask], data.y[data.train_mask]).item()
#     test_error = F.mse_loss(out[data.test_mask], data.y[data.test_mask]).item()
    
#     return train_error, test_error


def train():
    model.train()
    optimizer.zero_grad()

    # Get the model output
    out = model(data)

    # Ensure the output is between 0 and 1 by applying a sigmoid, if not already included in the model
    out_prob = torch.sigmoid(out[data.train_mask])

    # Ensure the target labels are of the correct shape [num_nodes, 1] and type float
    target = data.y[data.train_mask].unsqueeze(1).float()

    # Use binary cross-entropy loss
    loss = F.binary_cross_entropy(out_prob, target)

    # Perform backpropagation and an optimization step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()

    # Get the model output
    out = model(data)

    # Ensure the output is between 0 and 1 by applying a sigmoid
    out_prob = torch.sigmoid(out)

    # Calculate binary cross-entropy loss for train and test sets
    train_loss = F.binary_cross_entropy(out_prob[data.train_mask], data.y[data.train_mask].unsqueeze(1).float()).item()
    test_loss = F.binary_cross_entropy(out_prob[data.test_mask], data.y[data.test_mask].unsqueeze(1).float()).item()

    return train_loss, test_loss

# from torch_geometric.data import Batch

# def train():
#     model.train()
#     optimizer.zero_grad()

#     # Assuming 'devices' and 'data' are properly defined
#     data_list = [data.to(device) for device in devices]  # List of data objects for each GPU
#     out = model(data_list)  # Model should handle parallel execution and gathering

#     # Ensure outputs are gathered to a single device, typically cuda:0
#     out_prob = torch.sigmoid(out)  # This should be on a single device after gather
#     device = out_prob.device  # Device where the output is located

#     # Gather targets to the same device as out_prob
#     targets = torch.cat([d.y[d.train_mask].unsqueeze(1).float().to(device) for d in data_list])

#     # Compute loss
#     loss = F.binary_cross_entropy(out_prob, targets)
#     loss.backward()
#     optimizer.step()


# @torch.no_grad()
# def test():
#     model.eval()

#     # Get the model output
#     out = model(data)

#     # Apply sigmoid activation
#     out_prob = torch.sigmoid(out)

#     # Calculate binary cross-entropy loss for train and test sets
#     train_loss = F.binary_cross_entropy(out_prob[data.train_mask], data.y[data.train_mask].unsqueeze(1).float()).item()
#     test_loss = F.binary_cross_entropy(out_prob[data.test_mask], data.y[data.test_mask].unsqueeze(1).float()).item()

#     return train_loss, test_loss







# from torch_geometric.nn import GCNConv

# class ResidualGCNBlock(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, out_channels)
#         self.conv2 = GCNConv(out_channels, out_channels)
#         if in_channels != out_channels:
#             self.projection = torch.nn.Linear(in_channels, out_channels)
#         else:
#             self.projection = None

#     def forward(self, x, edge_index):
#         identity = x
#         out = F.relu(self.conv1(x, edge_index))
#         out = self.conv2(out, edge_index)
        
#         if self.projection is not None:
#             identity = self.projection(identity)
        
#         out += identity  # Residual connection
#         out = F.relu(out)
#         return out

# class ResidualGCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
#         super().__init__()
#         self.initial_conv = GCNConv(in_channels, hidden_channels)
#         self.res_blocks = torch.nn.ModuleList([ResidualGCNBlock(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
#         self.final_conv = GCNConv(hidden_channels, out_channels)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.initial_conv(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         for block in self.res_blocks:
#             x = block(x, edge_index)
#             x = F.dropout(x, p=self.dropout, training=self.training)
        
#         x = self.final_conv(x, edge_index)
#         return x

# def train(model, data, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)  # Shape should be [number of nodes, 1] for regression
#     out = out[data.train_mask]  # Applying mask
#     targets = data.y[data.train_mask].float()  # Ensure targets are float for MSE loss
#     loss = F.mse_loss(out, targets)  # Use MSE loss for continuous output
#     loss.backward()
#     optimizer.step()
#     return loss.item()

# def accuracy(output, labels, threshold=0.1):
#     """Calculate accuracy within a given threshold."""
#     return (output.sub(labels).abs() < threshold).float().mean().item()

# def evaluate(model, data):
#     model.eval()
#     with torch.no_grad():
#         out = model(data).squeeze()  # Squeeze to remove extra dimension from output

#         train_masked_out = out[data.train_mask]
#         test_masked_out = out[data.test_mask]
#         train_labels = data.y[data.train_mask].float()
#         test_labels = data.y[data.test_mask].float()

#         train_acc = accuracy(train_masked_out, train_labels)
#         test_acc = accuracy(test_masked_out, test_labels)

#         return train_acc, test_acc

# # Setup for model training/testing
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResidualGCN(data.num_node_features, 64, 1, num_layers=3).to(device)  # Assuming the output is 1 for regression
# data = data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # Lists to store training and test accuracy for each epoch
# train_acc_list = []
# test_acc_list = []

# # Example training loop
# for epoch in range(1, 101):
#     train_loss = train(model, data, optimizer)
#     train_acc, test_acc = evaluate(model, data)
#     train_acc_list.append(train_acc)
#     test_acc_list.append(test_acc)
#     print(f'Epoch {epoch}: Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

