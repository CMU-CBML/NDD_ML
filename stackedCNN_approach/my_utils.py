import os
import re
import numpy as np
# from collections import defaultdict
# import torch
# from torch_geometric.data import Data
import pandas as pd
# import random
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
    first_data = read_mesh_allCon(middle_file_path, 3)
    last_data = read_mesh_allCon(last_file_path, 3)

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

def read_mesh_cellCon(filename, grid_size=1.0, verbose=1):
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

# Define the required scalar fields
REQUIRED_FIELDS = ['phi', 'synaptogenesis', 'tubulin', 'tips', 'theta']

def read_mesh_cellCon_exception(filename, grid_size=1.0, verbose=1):
    """Reads and processes mesh data from a VTK file, ignoring z-coordinates and checking for NaN or INF values."""
    try:
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
            print(f"Warning: NaN or INF detected in point coordinates in file '{filename}'.")
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
                values = []
                for line in lines[lookup_table_start:lookup_table_start + num_points]:
                    try:
                        values.append(float(line.strip()))
                    except ValueError:
                        print(f"Warning: Invalid scalar value in field '{field_name}' in file '{filename}'.")
                        values.append(np.finfo(float).min)  # Assign a default value or handle as needed

                values = np.array(values)

                if any(np.isnan(values)) or any(np.isinf(values)):
                    print(f"Warning: NaN or INF detected in scalar values for field '{field_name}' in file '{filename}'.")
                    values = np.nan_to_num(values, nan=np.finfo(float).min, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
                scalar_fields[field_name] = values
                i = lookup_table_start + num_points
            else:
                i += 1

        # Check if all required fields are present
        missing_fields = [field for field in REQUIRED_FIELDS if field not in scalar_fields]
        if missing_fields:
            print(f"Error: Missing required fields {missing_fields} in file '{filename}'. Skipping this file.")
            return None  # Skip processing this file

        # Deduplicate scalar fields by taking the maximum value
        deduplicated_data = {name: np.full(len(unique_points), -np.inf, dtype=float) for name in REQUIRED_FIELDS}
        for name in REQUIRED_FIELDS:
            data = scalar_fields[name]
            for idx, value in zip(indices, data):
                if value > deduplicated_data[name][idx]:
                    deduplicated_data[name][idx] = value

        # Extract elements and compute edge attributes
        cells_start = lines.index(next(line for line in lines if 'CELLS' in line))
        num_cells = int(lines[cells_start].split()[1])
        cells_end = cells_start + 1 + num_cells
        elements = [list(map(int, line.strip().split()))[1:] for line in lines[cells_start + 1:cells_end]]
        elements = [indices[element] for element in elements]

        # Assuming edge_attributes need to be generated or are part of the dataset
        edge_attributes = {}  # This would need actual implementation

        if verbose:
            print(f"Processed scalar fields: {list(scalar_fields.keys())} in file '{filename}'.")
            print("Number of unique points:", len(unique_points))
            print("Sample deduplicated scalar data:", {k: v[:5] for k, v in deduplicated_data.items()})

        return unique_points, deduplicated_data, elements, pd.DataFrame(edge_attributes)
    
    except StopIteration:
        print(f"Error: 'POINTS' or 'CELLS' section not found in file '{filename}'. Skipping this file.")
        return None
    except Exception as e:
        print(f"Error processing file '{filename}': {e}. Skipping this file.")
        return None
    
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
    tree = KDTree(current_points)
    interpolated_data = {key: np.zeros(len(next_points)) for key in current_point_data.keys()}

    for i, point in enumerate(next_points):
        dists, indices = tree.query(point, k=k)  # Find k nearest neighbors
        
        if np.any(dists == 0):  # Check if any distance is zero
            # If a point is exactly at a data point, use the data from that point
            zero_dist_index = np.argmin(dists)
            for key in current_point_data.keys():
                interpolated_data[key][i] = current_point_data[key][indices[zero_dist_index]]
        else:
            weights = 1 / np.maximum(dists, 1e-6)  # Avoid division by zero
            weight_sum = np.sum(weights)
            
            for key in current_point_data.keys():
                feature_values = current_point_data[key][indices]
                interpolated_data[key][i] = np.dot(weights, feature_values) / weight_sum

    return interpolated_data

from scipy.spatial import cKDTree
import numpy as np

def interpolate_features_cKD(current_points, current_point_data, next_points, k=3):
    """
    Interpolate feature data from current points to next points using k-nearest neighbors,
    optimized with cKDTree and reducing loop overhead.

    Args:
        current_points (array): Coordinates of current points where data is known.
        current_point_data (dict): Dictionary mapping feature names to arrays of values.
        next_points (array): Coordinates of next points where data needs to be interpolated.
        k (int): Number of nearest neighbors to consider for interpolation.

    Returns:
        dict: A dictionary with interpolated features for each point in next_points.
    """
    tree = cKDTree(current_points)
    interpolated_data = {key: np.zeros(len(next_points)) for key in current_point_data.keys()}
    dists, indices = tree.query(next_points, k=k)  # Perform all queries at once

    # Iterate through keys to apply weights and compute interpolated values
    for key in current_point_data.keys():
        values = current_point_data[key]
        # Handle zero distances directly
        zero_dist_mask = (dists == 0)
        weights = np.where(zero_dist_mask, 1, 1 / np.maximum(dists, 1e-6))
        weight_sums = np.sum(weights, axis=1)

        for i, (weight, idx) in enumerate(zip(weights, indices)):
            if zero_dist_mask[i].any():
                interpolated_data[key][i] = values[idx[zero_dist_mask[i]][0]]
            else:
                interpolated_data[key][i] = np.dot(weight, values[idx]) / weight_sums[i]

    return interpolated_data
