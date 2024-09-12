import torch
from torch_spline_conv import spline_conv

x = torch.rand((4, 2), dtype=torch.float)  # 4 nodes with 2 features each
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])  # 6 edges
pseudo = torch.rand((6, 2), dtype=torch.float)  # two-dimensional edge attributes
weight = torch.rand((25, 2, 4), dtype=torch.float)  # 25 parameters for in_channels x out_channels
kernel_size = torch.tensor([5, 5])  # 5 parameters in each edge dimension
is_open_spline = torch.tensor([1, 1], dtype=torch.uint8)  # only use open B-splines
degree = 1  # B-spline degree of 1
norm = True  # Normalize output by node degree.
root_weight = torch.rand((2, 4), dtype=torch.float)  # separately weight root nodes
bias = None  # do not apply an additional bias

out = spline_conv(x, edge_index, pseudo, weight, kernel_size,
                  is_open_spline, degree, norm, root_weight, bias)

print(out.size())
torch.Size([4, 4])  # 4 nodes with 4 features each
