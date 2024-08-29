import os
import re
import numpy as np
from collections import defaultdict
import torch
from torch_geometric.data import Data
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Define a function to check if a point is on a line segment between two other points
def is_point_on_segment(p, q, r, tolerance=1e-6):
    """Check if point r lies on the line segment pq."""
    # Calculate vectors
    pq = q - p
    pr = r - p
    
    # Check if r is collinear with p and q
    collinear = np.isclose(np.cross(pq, pr), 0, atol=tolerance)
    if not collinear.all():  # Ensure all collinearity conditions are met
        return False
    
    # Check if r is within the bounds of the segment pq
    within_bounds = np.all(r >= np.minimum(p, q)) and np.all(r <= np.maximum(p, q))
    return within_bounds

# Define a function to hash points into a grid for spatial organization
def grid_hash(points, grid_size=1.0):
    """
    Hashes points into a spatial grid. Each point is assigned to a grid cell, facilitating
    quick spatial queries like proximity checks.

    Args:
        points (array): Array of point coordinates.
        grid_size (float): Dimension size of each grid cell.

    Returns:
        dict: A dictionary where keys are grid cell coordinates and values are lists of point indices in those cells.
    """
    grid = defaultdict(list)
    for idx, point in enumerate(points):
        # Calculate grid cell coordinates based on the point location and grid size.
        grid_key = (int(point[0] // grid_size), int(point[1] // grid_size))
        grid[grid_key].append(idx)
    return grid

def connect_points_in_zone(grid, grid_size=1.0):
    """
    Connects points within the same or adjacent grid cells to form edges, based on their spatial proximity.

    Args:
        grid (dict): A dictionary from grid_hash function mapping grid cells to point indices.
        grid_size (float): The size of each grid cell used in the hashing process.

    Returns:
        set: A set of tuples, each representing an edge between two points.
    """
    edges = set()
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for key in grid:
        for offset in neighbor_offsets:
            neighbor_key = (key[0] + offset[0], key[1] + offset[1])
            if neighbor_key in grid:
                # Check all points in the current cell against points in the neighboring cell
                for src in grid[key]:
                    for dest in grid[neighbor_key]:
                        if src != dest:
                            edges.add(tuple(sorted([src, dest])))
    return edges

def read_mesh_allCon(filename, grid_size=1.0):
    """
    Reads a VTK file and processes it to extract points, compute spatial connectivity,
    and extract scalar fields. Redefines connectivity based on spatial proximity.

    Args:
        filename (str): Path to the VTK file.
        grid_size (float): Size of the grid cell for spatial hashing.

    Returns:
        tuple: Contains arrays of unique points, deduplicated scalar data, edge list, and edge attributes.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Parse points
    points_start = lines.index(next(line for line in lines if 'POINTS' in line))
    num_points = int(lines[points_start].split()[1])
    points = np.array([list(map(float, line.strip().split()))[:2] for line in lines[points_start + 1:points_start + 1 + num_points]])

    # Remove duplicates and create spatial hash grid
    unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
    grid = grid_hash(unique_points, grid_size)

    # Compute spatially based edges
    edges = connect_points_in_zone(grid, grid_size)

    # Initialize scalar fields
    scalar_fields = {}
    i = points_start + num_points + 1
    while i < len(lines):
        if 'SCALARS' in lines[i] and 'LOOKUP_TABLE' in lines[i+1]:
            field_name = lines[i].split()[1]
            scalar_values = []
            start = i + 3
            end = min(start + num_points, len(lines))
            for j in range(start, end):
                try:
                    scalar_values.append(float(lines[j].strip()))
                except ValueError:
                    continue
            scalar_fields[field_name] = np.array(scalar_values)
            i = end
        else:
            i += 1

    # Map scalar values to unique points and take maximum values instead of averaging
    deduplicated_data = {name: np.full(len(unique_points), -np.inf) for name in scalar_fields}  # Initialize with negative infinity
    for name, data in scalar_fields.items():
        for idx, value in zip(inverse_indices, data):
            deduplicated_data[name][idx] = max(deduplicated_data[name][idx], value)  # Take max

    # Prepare edge attributes (e.g., distance)
    edge_attributes = [{'node1': a, 'node2': b, 'distance': np.linalg.norm(unique_points[a] - unique_points[b])} for a, b in edges]

    return unique_points, deduplicated_data, edges, edge_attributes

def sort_key_func(filename):
    """
    Extracts numbers from a filename and converts them to an integer for sorting.
    """
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')

def detailed_structure_info(data):
    print(f"List contains {len(data)} items.")
    for index, item in enumerate(data):
        if isinstance(item, np.ndarray):
            print(f"Item {index + 1}: Numpy array of shape {item.shape}, dtype {item.dtype}")
        elif isinstance(item, dict):
            print(f"Item {index + 1}: Dictionary with {len(item)} keys")
            print("   Keys:", list(item.keys()))
        elif isinstance(item, list):
            print(f"Item {index + 1}: List of {len(item)} items")
        elif isinstance(item, pd.DataFrame):
            print(f"Item {index + 1}: Pandas DataFrame with {item.shape[0]} rows and {item.shape[1]} columns")
            print("   Columns:", item.columns.tolist())
        else:
            print(f"Item {index + 1}: {type(item)}")

def read_vtk_file_pair(folder_path):
    """
    Reads the first and last VTK files in the specified folder that start with "physics_allparticles"
    and end with ".vtk", sorted by numeric order in filenames.

    Args:
        folder_path (str): Path to the folder containing VTK files.

    Returns:
        tuple: A tuple containing data from the middle and last VTK files.
    """
    # Filter files that start with "physics_allparticles" and end with ".vtk"
    vtk_files = [file for file in os.listdir(folder_path) 
                 if file.startswith("physics_allparticle") and file.endswith(".vtk")]
    
    vtk_files_sorted = sorted(vtk_files, key=sort_key_func)

    if not vtk_files_sorted:
        return None

    # Calculate the middle index
    middle_index = len(vtk_files_sorted) // 2

    # Read the middle and the last files
    middle_file_path = os.path.join(folder_path, vtk_files_sorted[middle_index])
    last_file_path = os.path.join(folder_path, vtk_files_sorted[-1])
    # last_file_path = os.path.join(folder_path, vtk_files_sorted[middle_index+2])

    # Assuming `read_mesh_allCon` is defined to read VTK files
    # first_data = read_mesh_allCon(middle_file_path, 3)
    # last_data = read_mesh_allCon(last_file_path, 3)
    first_data = read_mesh_cellCon(middle_file_path, 0)
    last_data = read_mesh_cellCon(last_file_path, 0)
    
    # print("++++++++++++++++++")
    # print(f"List contains {len(first_data)} items.")
    # print("Types of items:", [type(item) for item in first_data])
    # detailed_structure_info(first_data)
    # print("==================")
    return (first_data, last_data)

def read_all_folders_vtk_pairs(root_folder):
    """
    Reads the first and last VTK files from each subfolder within the root folder that start with "io2D".

    Args:
        root_folder (str): Path to the root folder containing subfolders with VTK files.

    Returns:
        list: A list of tuples, each containing data from the first and last VTK files from each subfolder.
    """
    folders = [f for f in os.listdir(root_folder) if f.startswith("io2D")]
    # folders = folders[:7] # for testing
    # print(folders)
    total_folders = len(folders)
    vtk_pairs = []
    
    for index, folder in enumerate(folders):
        outputs_path = os.path.join(root_folder, folder, "outputs")
        if os.path.isdir(outputs_path):
            vtk_pair = read_vtk_file_pair(outputs_path)
            if vtk_pair:
                vtk_pairs.append(vtk_pair)

        # Calculate and print the progress percentage
        progress_percent = ((index + 1) / total_folders) * 100
        print(f"Processing folder {index + 1}/{total_folders} ({progress_percent:.2f}%) completed: Reading from: {outputs_path}")

    return vtk_pairs

# Define a function to find points near a given line segment for further processing
def get_nearby_points(p, q, grid, points, grid_size=1.0):
    p_grid_key = (int(p[0] // grid_size), int(p[1] // grid_size))
    q_grid_key = (int(q[0] // grid_size), int(q[1] // grid_size))
    min_key = (min(p_grid_key[0], q_grid_key[0]), min(p_grid_key[1], q_grid_key[1]))
    max_key = (max(p_grid_key[0], q_grid_key[0]), max(p_grid_key[1], q_grid_key[1]))
    nearby_points = []
    # Collect all points within the bounding box of the segment pq
    for i in range(min_key[0], max_key[0] + 1):
        for j in range(min_key[1], max_key[1] + 1):
            if (i, j) in grid:
                nearby_points.extend(grid[(i, j)])
    return nearby_points

# Define a function to split edges based on the spatial proximity of nodes
def split_edge_by_nodes(edge, points, grid, grid_size=1.0):
    p, q = edge
    split_points = [p]
    nearby_points = get_nearby_points(points[p], points[q], grid, points, grid_size)
    for r in nearby_points:
        if is_point_on_segment(points[p], points[q], points[r]):
            split_points.append(r)
    split_points.append(q)
    split_points = sorted(set(split_points), key=lambda idx: np.linalg.norm(points[split_points[0]] - points[idx]))
    return [(split_points[i], split_points[i + 1]) for i in range(len(split_points) - 1)]

def read_mesh_cellCon(filename, verbose=1):
    """Reads and processes mesh data from a VTK file, ignoring z-coordinates and checking for NaN or INF values."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the start and end of the points section
    points_start = lines.index(next(line for line in lines if 'POINTS' in line))
    num_points = int(lines[points_start].split()[1])
    points_end = points_start + 1 + num_points

    # Extract points, ignoring z coordinates if present
    raw_points = [list(map(float, line.strip().split()))[:2] for line in lines[points_start + 1:points_end]]
    raw_points = np.array(raw_points)

    # Check for NaN or INF values in points
    if np.isnan(raw_points).any() or np.isinf(raw_points).any():
        print("Warning: NaN or INF detected in point coordinates.")
        # Optionally handle or filter these values
        raw_points = np.nan_to_num(raw_points, nan=np.finfo(float).min, posinf=np.finfo(float).max, neginf=np.finfo(float).min)

    unique_points, indices = np.unique(raw_points, axis=0, return_inverse=True)

    # Parse scalar fields and handle duplicates by taking the maximum value
    scalar_fields = {}
    i = points_end
    while i < len(lines):
        if 'SCALARS' in lines[i]:
            field_name = lines[i].split()[1]
            lookup_table_start = i + 2  # Points to the start of scalar values
            values = [float(line.strip()) for line in lines[lookup_table_start:lookup_table_start + num_points]]
            if any(np.isnan(values)) or any(np.isinf(values)):
                print(f"Warning: NaN or INF detected in scalar values for field '{field_name}'.")
                values = np.nan_to_num(values, nan=np.finfo(float).min, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
            scalar_fields[field_name] = np.array(values)
            i = lookup_table_start + num_points
        else:
            i += 1

    # Deduplicate scalar fields by taking the maximum value
    deduplicated_data = {name: np.full(len(unique_points), -np.inf, dtype=float) for name in scalar_fields}
    for name, data in scalar_fields.items():
        for i, idx in enumerate(indices):
            deduplicated_data[name][idx] = max(deduplicated_data[name][idx], data[i])

    # Extract elements and compute edge attributes
    cells_start = lines.index(next(line for line in lines if 'CELLS' in line))
    num_cells = int(lines[cells_start].split()[1])
    cells_end = cells_start + 1 + num_cells
    elements = [list(map(int, line.strip().split()))[1:] for line in lines[cells_start + 1:cells_end]]
    elements = [indices[element] for element in elements]

    # Assuming edge_attributes need to be generated or are part of the dataset
    edge_attributes = {}  # This would need actual implementation

    if verbose:
        print("Processed scalar fields:", list(scalar_fields.keys()))
        print("Number of unique points:", len(unique_points))
        print("Sample deduplicated scalar data:", {k: v[:5] for k, v in deduplicated_data.items()})

    return unique_points, deduplicated_data, elements, pd.DataFrame(edge_attributes)

def calculate_pseudo_coordinates(points, edge_index):
    """Calculate pseudo-coordinates for each edge based on node coordinates."""
    pseudo_coords = []
    for src, dest in edge_index.t().tolist():
        delta_x = points[dest, 0] - points[src, 0]
        delta_y = points[dest, 1] - points[src, 1]
        pseudo_coords.append([delta_x, delta_y])

    pseudo_coords = torch.tensor(pseudo_coords, dtype=torch.float)
    return pseudo_coords

def interpolate_features(current_points, current_point_data, next_points, k=3):
    """
    Interpolate feature data from current points to next points using k-nearest neighbors.

    Args:
        current_points (array): Coordinates of current points where data is known.
        current_point_data (dict): Dictionary mapping feature names to arrays of values.
        next_points (array): Coordinates of next points where data needs to be interpolated.
        k (int): Number of nearest neighbors to consider for interpolation.

    Returns:
        dict: A dictionary with interpolated features for each point in next_points.
    """
    # Create KDTree from current points
    tree = KDTree(current_points)
    interpolated_data = {key: np.zeros(len(next_points)) for key in current_point_data.keys()}

    # Iterate over each next point and interpolate features from nearest current points
    for i, point in enumerate(next_points):
        dists, indices = tree.query(point, k=k)  # Find k nearest neighbors
        
        # Handle cases where less than k points are available
        if not isinstance(indices, np.ndarray):
            indices = [indices]
            dists = [dists]

        weights = 1 / np.maximum(dists, 1e-6)  # Calculate weights inversely proportional to distance
        weight_sum = np.sum(weights)
        
        # Calculate weighted average for each feature
        for key in current_point_data.keys():
            # Extract the specific feature values for the nearest points
            feature_values = current_point_data[key][indices]
            # Compute the weighted average of the feature
            interpolated_data[key][i] = np.dot(weights, feature_values) / weight_sum

    return interpolated_data

def create_graph_data(points, point_data, edge_attributes, y_values):
    """
    Create graph data from provided points, elements, point data, edge attributes, and target labels.

    Args:
        points (array): Coordinates of points.
        point_data (dict): Features for each point, expected to be a dict of arrays.
        edge_attributes (list or DataFrame): Attributes for edges.
        y_values (array): Target labels for each point.
    """
    # Convert points to tensor and ensure type is float
    points_tensor = torch.tensor(points, dtype=torch.float)

    # Prepare features tensor by concatenating feature arrays stored in point_data dictionary
    feature_tensors = [torch.tensor(point_data[key], dtype=torch.float).unsqueeze(1) for key in point_data.keys()]
    point_features = torch.cat([points_tensor] + feature_tensors, dim=1)

    # Prepare targets using and next_phi
    y = torch.tensor(y_values, dtype=torch.float)  # Assuming y_values are directly passable to tensor creation

    # Convert edge attributes to tensor
    if isinstance(edge_attributes, list):
        edge_attributes = pd.DataFrame(edge_attributes)
    edge_index = torch.tensor(edge_attributes[['node1', 'node2']].to_numpy().T, dtype=torch.long)

    # Calculate pseudo-coordinates for edge attributes if needed
    pseudo_coords = calculate_pseudo_coordinates(points_tensor, edge_index)
    
    # Construct graph data object
    data = Data(x=point_features, edge_index=edge_index, edge_attr=pseudo_coords, y=y)
    return data

def create_graphs_from_datasets(vtk_pairs):
    """
    Processes pairs of VTK data, where each pair's first item is current data and the second item is next data.

    Args:
        vtk_pairs (list of tuples): List where each tuple contains two sets of data (current and next).

    Returns:
        list: A list of graph data objects, each representing processed graph data from the pairs.
    """
    graph_data_list = []

    for current_data, next_data in vtk_pairs:
        current_points, current_point_data, _, _ = current_data
        next_points, next_point_data, _, next_edge_attributes = next_data

        # Interpolate current point data to next points
        interpolated_point_data = interpolate_features(current_points, current_point_data, next_points)
        interpolated_point_data['theta'] = next_point_data['theta']
        
        # # Y values are directly the phi values from next points
        # y_values = next_point_data['phi']

        # Assuming interpolated_point_data and next_point_data are defined and contain 'phi'
        y_values = np.round(interpolated_point_data['phi']) - np.round(next_point_data['phi'])
        # Keeping only positive values
        y_values = np.maximum(y_values, 0)
        # Rounding to 1 decimal place
        y_values = np.round(y_values, 1)
        
        # Create the graph data using next points, interpolated data, and attributes
        data = create_graph_data(next_points, interpolated_point_data, next_edge_attributes, y_values)
        graph_data_list.append(data)

        progress_percent = (len(graph_data_list) / len(vtk_pairs)) * 100
        print(f"Processing pair {len(graph_data_list)} of {len(vtk_pairs)} ({progress_percent:.2f}%) completed")

    return graph_data_list

def plot_graph_components_with_highlights(points, features, edges, edge_attr, y):
    """
    Plots nodes and multiple scalar values in subplots, highlighting in blue the vectors (edges) originating from a randomly selected node,
    based on edge_index, using edge_attr to determine direction and magnitude, and also plots the y-values for each node.

    Args:
        points (numpy.ndarray): Nx2 array of point coordinates.
        features (numpy.ndarray): Array of multiple scalar values associated with each point, dimension NxM.
        edges (numpy.ndarray): Nx2 array of index pairs representing edges.
        edge_attr (torch.Tensor): Tensor containing lengths or vector components for each edge.
        y (numpy.ndarray): Array of scalar values associated with each point, typically used as target labels or another feature.
    """
    random_node = random.randint(0, len(points) - 1)
    num_features = features.shape[1]

    # Layout setup
    num_columns = 3
    num_rows = ((num_features + 2) + num_columns - 1) // num_columns
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 4 * num_rows))
    axs = axs.ravel()

    # Plot node features
    for i in range(num_features):
        scatter = axs[i].scatter(points[:, 0], points[:, 1], c=features[:, i], cmap='viridis', s=20, alpha=0.6, zorder=1)
        axs[i].set_title(f'Feature {i+1}')
        fig.colorbar(scatter, ax=axs[i])

    # Plot y values
    scatter_y = axs[num_features].scatter(points[:, 0], points[:, 1], c=y, cmap='viridis', s=20, alpha=0.6, zorder=1)
    axs[num_features].set_title('Node Distribution with Y Values')
    fig.colorbar(scatter_y, ax=axs[num_features])

    # Prepare to plot edges
    edge_ax = axs[num_features + 1]
    edge_ax.scatter(points[:, 0], points[:, 1], c='lightgray', s=10, alpha=0.5, zorder=1)

    # Plot all edges with lower zorder
    for index, edge in enumerate(edges):
        point_a = points[edge[0]]
        point_b = points[edge[1]]
        dx = edge_attr[index][0]
        dy = edge_attr[index][1]
        edge_ax.plot([point_a[0], point_a[0]+dx], [point_a[1], point_a[1]+dy], color='gray', alpha=0.3, zorder=2, linewidth=1)

    # Plot connected edges last with higher zorder
    for index, edge in enumerate(edges):
        if random_node in edge:
            point_a = points[edge[0]]
            point_b = points[edge[1]]
            dx = edge_attr[index][0]
            dy = edge_attr[index][1]
            edge_ax.plot([point_a[0], point_a[0]+dx], [point_a[1], point_a[1]+dy], color='blue', zorder=3, linewidth=2)

    edge_ax.set_title(f'Edges Highlighting Node {random_node}')
    edge_ax.scatter(points[random_node, 0], points[random_node, 1], color='red', s=30, label='Random Node', zorder=4)
    # edge_ax.legend()

    # Hide any unused axes
    total_plots = num_features + 2
    for i in range(total_plots, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    
def combine_graph_data(graph_data_list):
    """
    Combines multiple graph data objects into a single graph data object.

    Args:
        graph_data_list (list of Data): List of individual graph data objects to be combined.

    Returns:
        Data: A single combined graph data object containing all node features, edges, edge attributes, and target labels.
    """
    # Initialize lists to hold combined data components
    combined_x = []
    combined_edge_index = []
    combined_edge_attr = []
    combined_y = []
    
    node_offset = 0  # Initial offset for node indices

    # Loop through each individual graph data object
    for data in graph_data_list:
        num_nodes = data.x.size(0)  # Number of nodes in the current graph
        
        # Append node features from the current graph
        combined_x.append(data.x)
        
        # Adjust edge indices by the current node offset and append
        adjusted_edge_index = data.edge_index + node_offset
        combined_edge_index.append(adjusted_edge_index)
        
        # Append edge attributes from the current graph
        combined_edge_attr.append(data.edge_attr)
        
        # Append target labels from the current graph
        combined_y.append(data.y)
        
        # Update the node offset for the next graph
        node_offset += num_nodes
    
    # Concatenate lists into tensors to create a single combined graph
    combined_x = torch.cat(combined_x, dim=0)
    combined_edge_index = torch.cat(combined_edge_index, dim=1)
    combined_edge_attr = torch.cat(combined_edge_attr, dim=0)
    combined_y = torch.cat(combined_y, dim=0)
    
    # Create and return the combined Data object
    combined_data = Data(x=combined_x, edge_index=combined_edge_index, edge_attr=combined_edge_attr, y=combined_y)
    return combined_data

def remove_nodes(data, threshold=1e-1):
    """
    Removes nodes based on the condition that the third channel of x for the node and all its connected nodes 
    is close to 0 or close to 1 within a threshold.

    Args:
        data (Data): PyTorch Geometric Data object with attributes x, edge_index, y, and optionally edge_attr.
        threshold (float): Threshold value to determine closeness to zero or one.

    Returns:
        Data: A new PyTorch Geometric Data object with specified nodes removed.
    """
    num_nodes = data.x.size(0)
    # Create an adjacency list to find connected nodes easily
    adjacency_list = [[] for _ in range(num_nodes)]
    for source, target in data.edge_index.t().tolist():
        adjacency_list[source].append(target)
        adjacency_list[target].append(source)  # Assuming undirected graph

    # Check each node and its neighbors
    keep_nodes = torch.ones(num_nodes, dtype=torch.bool)
    phi_values = data.x[:, 2]  # Assume the 'phi' feature is the third column in x
    for idx in range(num_nodes):
        phi_node = phi_values[idx]
        neighbors_phi = phi_values[adjacency_list[idx]]
        # Check if node and all connected nodes are close to 0 or 1
        if not ((torch.abs(phi_node) <= threshold).item() or (torch.abs(phi_node - 1) <= threshold).item()):
            continue  # If central node phi is not close to 0 or 1, keep it
        if torch.all(torch.abs(neighbors_phi) <= threshold) or torch.all(torch.abs(neighbors_phi - 1) <= threshold):
            keep_nodes[idx] = False  # If all neighbors are also close to 0 or 1, mark for removal

    # Filter nodes, edges, and edge attributes
    new_x = data.x[keep_nodes]
    new_y = data.y[keep_nodes]
    kept_node_indices = torch.where(keep_nodes)[0]
    new_indices = torch.full((num_nodes,), -1, dtype=torch.long)
    new_indices[kept_node_indices] = torch.arange(kept_node_indices.size(0))

    # Filter edges to include only those connecting kept nodes
    edge_mask = (new_indices[data.edge_index[0]] != -1) & (new_indices[data.edge_index[1]] != -1)
    new_edge_index = new_indices[data.edge_index[:, edge_mask]]

    new_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None

    # Create a new data object with the filtered data
    new_data = Data(x=new_x, edge_index=new_edge_index, y=new_y, edge_attr=new_edge_attr)
    return new_data

def remove_nans_and_infs(data):
    # Verify and convert data types for NaN and Inf support if necessary
    data.x = data.x.float() if not data.x.is_floating_point() else data.x
    data.y = data.y.float() if not data.y.is_floating_point() else data.y

    # Check for NaNs and Infs in node features
    nan_inf_nodes_x = torch.isnan(data.x).any(dim=1) | torch.isinf(data.x).any(dim=1)
    print(f"NaNs or Infs in node features: {nan_inf_nodes_x.sum().item()} found.")

    # Check for NaNs and Infs in labels, supports multidimensional labels
    if data.y.dim() > 1:
        nan_inf_nodes_y = torch.isnan(data.y).any(dim=1) | torch.isinf(data.y).any(dim=1)
    else:
        nan_inf_nodes_y = torch.isnan(data.y) | torch.isinf(data.y)
    print(f"NaNs or Infs in labels: {nan_inf_nodes_y.sum().item()} found.")

    # Combine conditions to identify all nodes with any NaN or Inf
    nan_inf_nodes = nan_inf_nodes_x | nan_inf_nodes_y

    if nan_inf_nodes.any():
        print("NaNs or Infs found in the dataset. Removing affected nodes.")
    else:
        print("No NaNs or Infs detected in the nodes.")

    # Filter out nodes with NaNs or Infs
    valid_nodes_indices = (~nan_inf_nodes).nonzero(as_tuple=True)[0]
    data.x = data.x[valid_nodes_indices]
    data.y = data.y[valid_nodes_indices]

    # Update edge_index if present, removing edges connected to NaN or Inf nodes
    if hasattr(data, 'edge_index'):
        edge_index = data.edge_index
        # Filter edges to remove any referencing removed nodes
        mask = (~nan_inf_nodes[edge_index[0]]) & (~nan_inf_nodes[edge_index[1]])
        filtered_edge_index = edge_index[:, mask]

        # Update edge indices to account for removed nodes
        new_index = torch.full((len(nan_inf_nodes),), -1, dtype=torch.long)
        new_index[valid_nodes_indices] = torch.arange(len(valid_nodes_indices), device=edge_index.device)
        data.edge_index = new_index[filtered_edge_index]

    return data

def min_max_normalize_features(x):
    """
    Normalizes each feature in the x tensor to be between 0 and 1.

    Args:
        x (torch.Tensor): A tensor of shape [N, F] where N is the number of nodes and F is the number of features.

    Returns:
        torch.Tensor: The normalized feature tensor.
    """
    min_vals = torch.min(x, dim=0)[0]
    max_vals = torch.max(x, dim=0)[0]
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Prevent division by zero to handle features that are constant
    print(min_vals)
    print(max_vals)
    print(range_vals)
    normalized_x = (x - min_vals) / range_vals

    return normalized_x

def add_gradient_features(data, edge_index):
    """
    Adds gradient features to the data.x tensor based on the differences in feature values along edges.
    The gradient is calculated using the pseudo-coordinates stored in edge_attr and added to the node features.

    Args:
        data (torch_geometric.data.Data): The data object containing node features and edge attributes.
        edge_index (torch.Tensor): Tensor containing the indices of source and destination nodes for each edge.

    Returns:
        data (torch_geometric.data.Data): The modified data object with additional gradient features.
    """
    num_features = data.x.shape[1] - 2  # Exclude the coordinate columns
    epsilon = 1e-6  # Small number to prevent division by zero

    for i in range(num_features):
        feature_values = data.x[:, i+2]
        gradients = torch.zeros_like(data.x[:, :2])  # Only two columns for gradient (dx, dy)

        for j, (src, dest) in enumerate(edge_index.t()):
            pseudo_coords = data.edge_attr[j]
            dx, dy = pseudo_coords[0], pseudo_coords[1]  # Assuming pseudo_coords are stored as [dx, dy]

            # Calculate gradient components separately and safely
            gradient_x = (feature_values[dest] - feature_values[src]) / (dx + epsilon)
            gradient_y = (feature_values[dest] - feature_values[src]) / (dy + epsilon)
            
            gradients[src, 0] += gradient_x
            gradients[src, 1] += gradient_y
            gradients[dest, 0] -= gradient_x
            gradients[dest, 1] -= gradient_y  # Symmetric impact

        # Concatenate gradient features to original features
        data.x = torch.cat([data.x, gradients], dim=1)

        print(f"Processed feature {i+1}/{num_features} ({(i+1) / num_features * 100:.1f}%)")

    return data

def rotate_points(data, angle_degrees):
    """
    Rotates the point coordinates in the data.x tensor by a specified angle.

    Args:
        data (torch_geometric.data.Data): The data object containing node features.
        angle_degrees (float): The angle in degrees to rotate the point coordinates.

    Returns:
        data (torch_geometric.data.Data): The modified data object with rotated coordinates.
    """
    angle = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # Assume the first two columns in x are coordinates
    coords = data.x[:, :2].numpy()
    new_coords = np.dot(coords, rotation_matrix)
    data.x[:, :2] = torch.tensor(new_coords, dtype=torch.float32)

    print(f"Rotated points by {angle_degrees} degrees.")
    return data

def oversample_minority_class(data):
    """
    Duplicates samples of the minority class in the dataset to balance class distribution.

    Args:
        data (torch_geometric.data.Data): The data object containing node labels and optionally masks.

    Returns:
        data (torch_geometric.data.Data): The modified data object with duplicated minority class samples.
    """
    unique, counts = torch.unique(data.y, return_counts=True)
    minority_class = unique[torch.argmin(counts)]

    # Indices of the minority class
    minority_indices = (data.y == minority_class).nonzero(as_tuple=True)[0]
    
    # Duplicate minority class samples
    minority_x = data.x[minority_indices]
    minority_y = data.y[minority_indices]
    
    # Update the data object
    data.x = torch.cat([data.x, minority_x], dim=0)
    data.y = torch.cat([data.y, minority_y], dim=0)

    # Update masks if they exist
    if hasattr(data, 'train_mask'):
        minority_mask = data.train_mask[minority_indices]
        data.train_mask = torch.cat([data.train_mask, minority_mask], dim=0)
    if hasattr(data, 'test_mask'):
        minority_mask = data.test_mask[minority_indices]
        data.test_mask = torch.cat([data.test_mask, minority_mask], dim=0)
    if hasattr(data, 'val_mask'):
        minority_mask = data.val_mask[minority_indices]
        data.val_mask = torch.cat([data.val_mask, minority_mask], dim=0)
    
    print(f"Duplicating node {idx.item()} for minority class ({minority_class}); Progress: {len(data.y) / len(minority_indices) * 100:.1f}%")
    return data

def plot_features_and_target(data):
    """
    Plots each feature and the target variable from the dataset based on the coordinates.

    Args:
        data (torch_geometric.data.Data): The data object containing node features, where
            the first two columns are assumed to be x and y coordinates.

    Displays:
        Scatter plots for each feature and the target variable, colored by their values.
    """
    if data.x is None or data.y is None:
        print("Node features or target values are missing in the dataset.")
        return

    # Coordinates are the first two columns
    coords = data.x[:, :2].numpy()
    num_features = data.x.size(1) - 2  # excluding the coordinate columns

    # Setup plot grid
    num_plots = num_features + 1  # Plus one for the target variable
    cols = 3  # Set number of columns for subplots
    rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharey=True, squeeze=False)

    # Plot each feature
    for i in range(num_plots):
        ax = axes[i // cols, i % cols]  # Determine row and column index
        if i < num_features:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=data.x[:, i+2].numpy(), cmap='viridis', s=35)
            fig.colorbar(sc, ax=ax, label=f'Feature {i+2}')
            ax.set_title(f'Feature {i+2} Distribution')
        else:
            # Plot the target variable
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=data.y.numpy(), cmap='viridis', s=35)
            fig.colorbar(sc, ax=ax, label='Target')
            ax.set_title('Target Distribution')

        ax.set_xlabel('X Coordinate')
        if i % cols == 0:  # Only set y-label on the first column
            ax.set_ylabel('Y Coordinate')

    # Hide empty subplots if any
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])

    plt.tight_layout()
    plt.show()
    
def print_random_y_values(data):
    """
    Randomly selects and prints 10 values from the data.y tensor.

    Args:
        data (torch_geometric.data.Data): The data object containing the y attribute
            which is a tensor of shape [N] where N is the number of elements.

    """
    num_samples = 20  # Number of samples to print
    num_elements = data.y.size(0)  # Total number of elements in data.y

    # Ensure we don't sample more elements than available
    if num_elements < num_samples:
        num_samples = num_elements

    # Randomly choose indices without replacement
    indices = torch.randperm(num_elements)[:num_samples]

    # Fetch the selected elements
    selected_values = data.y[indices]

    # Print the selected values
    print("Randomly selected data.y values:")
    for index, value in zip(indices, selected_values):
        print(f'Index {index.item()}: {value.item()}')

def check_tensor(tensor, name="Tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"{name} contains NaN or Inf.")
