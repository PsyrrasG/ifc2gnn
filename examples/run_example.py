import sys
sys.path.append("../ifc2gnn")

from ifc2gnn.ifc2gnn import IFCStructuralGraphParser

# Load and parse the IFC file
parser = IFCStructuralGraphParser("./examples/building_1_S_schema2x3.ifc")
graph = parser.parse()

# Get adjacency matrix
adj_matrix = parser.get_adjacency_matrix()
print("Adjacency matrix shape:", adj_matrix.shape)
print("Adjacency matrix:\n", adj_matrix)

# Convert to PyTorch Geometric
pyg_data = parser.to_pytorch_geometric()
print("PyG data:", pyg_data)

print("Graph Statistics:")
print(f"Number of connection points (nodes): {pyg_data.num_nodes}")
print(f"Number of structural members (edges): {pyg_data.num_edges}")
print(f"Node features shape: {pyg_data.x.shape}")
print(f"Edge features shape: {pyg_data.edge_attr.shape}")

# Visualize
parser.visualize()

# Example: Accessing node positions (3D coordinates)
print("\nFirst 5 node positions:")
print(pyg_data.pos[:5])

# Example: Accessing edge features
print("\nFirst 5 edge features:")
print(pyg_data.edge_attr[:5])