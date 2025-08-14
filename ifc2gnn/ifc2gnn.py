import ifcopenshell
import ifcopenshell.geom
from ifcopenshell.util import placement as ifc_placement
import ifcopenshell.util.element
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class IFCStructuralGraphParser:
    """
    A parser that converts IFC (structural analysis view) files into graphs where:
    - Nodes represent connection points (joints, supports)
    - Edges represent structural elements (beams, columns)
    
    Args:
        file_path (str): Path to the IFC file
        include_properties (bool): Whether to include material/section properties
        include_non_structural (bool): Whether to include non-structural elements
    """
    
    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.include_properties = kwargs.get('include_properties', True)
        self.include_non_structural = kwargs.get('include_non_structural', False)
        
        # Geometry settings
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)
        self.settings.set(self.settings.INCLUDE_CURVES, True)
        
        try:
            self.ifc_file = ifcopenshell.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open IFC file: {str(e)}")
        
        self.graph = None
        self.node_mapping = {}
        self.element_mapping = {}
        self.node_counter = 0
        self.property_sets = [
            'Pset_ElementMaterial', 
            'Pset_ProfileMechanical',
            'Pset_StructuralLoad',
            'Pset_StructuralAnalysis'
        ]
        
    def parse(self) -> nx.Graph:
        """Parse the IFC file and construct the connection-point graph"""
        self.ifc_file = ifcopenshell.open(self.file_path)
        self.graph = nx.Graph()
        
        # Step 1: Identify all connection points (nodes)
        self._identify_connection_points()
        
        # Step 2: Process structural elements (edges)
        self._identify_structural_elements()
        
        return self.graph
    
    # JOINTS/NODES:
    # -------------
    def _identify_connection_points(self):
        """Identify all connection points that will become graph nodes"""
        # Get explicit structural connection points
        point_connections = self.ifc_file.by_type('IfcStructuralPointConnection')
        
        # Get structural curve connections (might represent supports)
        curve_connections = self.ifc_file.by_type('IfcStructuralCurveConnection')
        
        # Get supports (base connections)
        supports = self.ifc_file.by_type('IfcStructuralPointReaction')
        
        # Collect all potential connection points
        all_connections = point_connections + curve_connections + supports
        
        # Create nodes for each connection point
        for connection in all_connections:
            self._add_connection_node(connection)
    

    def _add_connection_node(self, connection) -> int:
        """Add a connection point as a graph node and return its ID"""
        node_id = self.node_counter
        self.node_mapping[connection.id()] = node_id
        
        # Get connection point coordinates
        placement = self._get_connection_coordinates(connection)
        
        # Basic node attributes
        attributes = {
            'ifc_type': connection.is_a(),
            'global_id': connection.GlobalId,
            'name': getattr(connection, 'Name', ''),
            'coordinates': placement,
            'is_support': connection.is_a('IfcStructuralPointReaction')
        }
        
        if self.include_properties:
            # Add connection properties
            attributes.update(self._get_connection_properties(connection))
        
        self.graph.add_node(node_id, **attributes)
        self.node_counter += 1
        return node_id
    

    def _get_connection_properties(self, connection) -> Dict:
        """Extract properties from a connection point"""
        properties = {}

        # Always include coordinates
        properties['coordinates'] = self._get_connection_coordinates(connection)
        
        # Fixity conditions with None handling
        condition = getattr(connection, 'AppliedCondition', None)
        if condition:
            properties.update({
                'fixity_x': getattr(condition, 'TranslationalStiffnessX', None),
                'fixity_y': getattr(condition, 'TranslationalStiffnessY', None),
                'fixity_z': getattr(condition, 'TranslationalStiffnessZ', None),
                'rotational_fixity_x': getattr(condition, 'RotationalStiffnessX', None),
                'rotational_fixity_y': getattr(condition, 'RotationalStiffnessY', None),
                'rotational_fixity_z': getattr(condition, 'RotationalStiffnessZ', None)
            })
        
        return properties
    

    def _get_connection_coordinates(self, connection) -> Tuple[float, float, float]:
        """
        Get coordinates from connection point using index path (explicit).
        connection: "IfcStructuralPointConnection"
        [6] -> "IfcProductDefinitionShape"
        [2][0] -> "IfcTopologyRepresentation"
        [6][2][0][3][0] -> "IfcVertexPoint"
        [6][2][0][3][0][0] -> "IfcCartesianPoint"
        """
        try:
            coord_tuple = connection[6][2][0][3][0][0][0]
            return coord_tuple
            # placement = ifc_placement.get_local_placement(connection.ObjectPlacement)
            # if placement is not None:
            #     return (float(placement[0, 3]),
            #             float(placement[1, 3]),
            #             float(placement[2, 3]))
        except Exception as e:
            print(f"Coordinate extraction warning for {connection.id()}: {str(e)}")
            return (0.0, 0.0, 0.0)
        

    # ELEMENTS/MEMBERS:
    # ---------------
    def _identify_structural_elements(self):
        """
        Process structural elements to create edges between connection points
        An 'IfcProduct' can be either: 
        IfcStructuralItem -> IfcStructuralMember -> IfcStructuralCurveMember [structural member]
        or
        IfcElement -> IfcBuildingElementn -> Ifcbeam etc. [may or not be structural]
        """
        # Get all structural members (beams, columns, etc.)
        members = self.ifc_file.by_type('IfcStructuralMember')
        
        # Also include standard beams and columns if non-structural included
        if self.include_non_structural:
            members += self.ifc_file.by_type('IfcBeam') + self.ifc_file.by_type('IfcColumn')
        
        for member in members:
            self._process_structural_member(member)
    

    def _process_structural_member(self, member):
        """Process a single structural member and create corresponding edge"""
        # Get connections for this member
        connections = self._get_member_connections(member)
        
        # We need exactly 2 connections for beams/columns
        if len(connections) == 2:
            node1 = self.node_mapping.get(connections[0].id())
            node2 = self.node_mapping.get(connections[1].id())
            
            if node1 is not None and node2 is not None:
                # Create edge with member properties
                edge_attrs = self._get_member_properties(member)
                edge_attrs['member_type'] = member.is_a()
                edge_attrs['global_id'] = member.GlobalId
                
                self.graph.add_edge(node1, node2, **edge_attrs)
                self.element_mapping[member.id()] = (node1, node2)
    

    def _get_member_connections(self, member) -> List:
        """Get connection points for a structural member"""
        connections = []
        
        # Check for explicit structural connections
        for rel in member.ConnectedBy or []:
            if rel.is_a('IfcRelConnectsStructuralMember'):
                connections.append(rel.RelatedStructuralConnection)
        
        # Fallback to geometric connections if no structural connections found
        if not connections and hasattr(member, 'ConnectedTo'):
            for rel in member.ConnectedTo:
                if rel.is_a('IfcRelConnectsElements'):
                    connections.append(rel.RelatedElement)
        
        return connections


    def _get_member_properties(self, member) -> Dict:
        """Extract properties from a structural member with schema compatibility"""
        properties = {
            'member_type': member.is_a(),
            'global_id': getattr(member, 'GlobalId', ''),
            'name': getattr(member, 'Name', '')
        }

        # Material properties with schema checks
        material = ifcopenshell.util.element.get_material(member)
        if material:
            if material.is_a('IfcMaterial'):
                properties.update(self._get_material_properties(material))
            elif material.is_a('IfcMaterialProfileSet'):
                properties.update(self._get_profile_set_properties(material))
            elif material.is_a('IfcMaterialLayerSet'):  # For wall-like elements
                properties.update(self._get_layer_set_properties(material))

        # Geometric properties
        properties['length'] = self._get_member_length(member)
        
        # Property sets with fallbacks
        psets = ifcopenshell.util.element.get_psets(member)
        for pset_name in self.property_sets:
            if pset_name in psets:
                for prop_name, prop_value in psets[pset_name].items():
                    # Handle both IFC2X3 and IFC4 property value formats
                    if isinstance(prop_value, dict) and 'wrappedValue' in prop_value:
                        properties[prop_name] = prop_value['wrappedValue']
                    else:
                        properties[prop_name] = prop_value
        
        # Schema-specific fallbacks
        if self.ifc_file.schema == 'IFC2X3':
            if 'CrossSectionArea' not in properties:
                properties['CrossSectionArea'] = getattr(
                    ifcopenshell.util.element.get_psets(member).get('ProfileMechanical', {}), 
                    'CrossSectionArea', 0.0
                )
        
        return properties
    

    def _get_layer_set_properties(self, layer_set):
        """Handle material layer sets (for walls, slabs, etc.)"""
        properties = {}
        if hasattr(layer_set, 'MaterialLayers'):
            layers = layer_set.MaterialLayers or []
            if layers:
                # Just take the first layer's properties
                layer = layers[0]
                if hasattr(layer, 'Material'):
                    properties.update(self._get_material_properties(layer.Material))
                properties['layer_thickness'] = getattr(layer, 'LayerThickness', 0.0)
        return properties
    

    def _get_material_properties(self, material):
        """Get material properties compatible with multiple IFC schemas"""
        props = {}
        
        # Common properties across schemas
        props['name'] = getattr(material, 'Name', '')
        
        # Handle different schema versions
        if hasattr(material, 'Category'):  # IFC4 and later
            props['category'] = material.Category
        else:  # IFC2X3
            props['category'] = ''
            
        # Additional properties for material definitions
        if hasattr(material, 'HasProperties'):
            for prop in getattr(material, 'HasProperties', []):
                if prop.is_a('IfcMaterialProperties'):
                    for single_prop in getattr(prop, 'Properties', []):
                        props[single_prop.Name.lower()] = single_prop.NominalValue.wrappedValue
        
        return props
    

    def _get_profile_set_properties(self, profile_set):
        """Get properties from material profile set"""
        properties = {}
        
        if hasattr(profile_set, 'MaterialProfiles'):
            profiles = profile_set.MaterialProfiles or []
            if profiles:
                profile = profiles[0]
                if hasattr(profile, 'Profile'):
                    prof = profile.Profile
                    properties['profile_type'] = getattr(prof, 'ProfileType', '')
                    properties['profile_name'] = getattr(prof, 'ProfileName', '')
                    
                    # Cross-section properties
                    properties['area'] = getattr(prof, 'CrossSectionArea', 0.0)
                    properties['inertia_y'] = getattr(prof, 'MomentOfInertiaY', 0.0)
                    properties['inertia_z'] = getattr(prof, 'MomentOfInertiaZ', 0.0)
        
        return properties
    
    
    def _get_member_length(self, member) -> float:
        """Calculate member length using geometry"""
        try:
            # Try to get length from geometry first
            shape = ifcopenshell.geom.create_shape(self.settings, member)
            if shape:
                verts = shape.geometry.verts
                if len(verts) >= 6:  # At least start and end points
                    start = np.array(verts[:3])
                    end = np.array(verts[-3:])
                    return float(np.linalg.norm(end - start))
        except Exception as e:
            print(f"Geometry length warning for {member.id()}: {str(e)}")
        
        # Fallback to connection distance
        connections = self._get_member_connections(member)
        if len(connections) == 2:
            coord1 = self._get_connection_coordinates(connections[0])
            coord2 = self._get_connection_coordinates(connections[1])
            return float(np.linalg.norm(np.array(coord2) - np.array(coord1)))
        
        return 0.0


    # PREPARE FOR USE IN PYTORCH
    def to_pytorch_geometric(self) -> Data:
        """
        Convert the networkx graph to a PyTorch Geometric Data object.
        Returns:
            Data: A PyTorch Geometric Data object ready for GNN processing
        """
        if self.graph is None:
            raise ValueError("Graph not created yet. Call parse() first.")
        
        # Node features (connection points/joints)
        node_features = []
        node_coordinates = []
        for node_id, node_data in self.graph.nodes(data=True):
            features = self._extract_node_features(node_data)
            node_features.append(features)
            node_coordinates.append(node_data.get('coordinates', [0, 0, 0]))
        
        # Edge features and indices (structural members/elements)
        edge_indices = []
        edge_features = []
        for src, dst, edge_data in self.graph.edges(data=True):
            edge_indices.append([src, dst])
            features = self._extract_edge_features(edge_data)
            edge_features.append(features)
        
        # Convert to tensors
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        pos = torch.tensor(np.array(node_coordinates), dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
        
        return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the structural graph.
        Returns: Numpy array where 1 indicates a connection between nodes
        """
        return nx.to_numpy_array(self.graph)


    def _extract_node_features(self, node_data: Dict) -> np.ndarray:
        """Convert node attributes to a feature vector with robust None handling"""
        features = []
        
        # Connection type features (one-hot encoded style)
        ifc_type = node_data.get('ifc_type', '')
        features.extend([
            1.0 if 'PointConnection' in ifc_type else 0.0,
            1.0 if 'CurveConnection' in ifc_type else 0.0,
            1.0 if node_data.get('is_support', False) else 0.0
        ])
        
        # Fixity conditions with None handling
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default
        
        features.extend([
            safe_float(node_data.get('fixity_x')),
            safe_float(node_data.get('fixity_y')),
            safe_float(node_data.get('fixity_z')),
            safe_float(node_data.get('rotational_fixity_x')),
            safe_float(node_data.get('rotational_fixity_y')),
            safe_float(node_data.get('rotational_fixity_z'))
        ])
        
        # Coordinate features (always available)
        coordinates = node_data.get('coordinates', (0.0, 0.0, 0.0))
        features.extend([
            safe_float(coordinates[0]),
            safe_float(coordinates[1]),
            safe_float(coordinates[2])
        ])
        
        return np.array(features)
    

    def _extract_edge_features(self, edge_data: Dict) -> np.ndarray:
        """Convert edge attributes to a feature vector"""
        features = []
        
        # Member type
        member_type = edge_data.get('member_type', '')
        features.extend([
            1 if 'Beam' in member_type else 0,
            1 if 'Column' in member_type else 0,
            1 if 'Member' in member_type else 0
        ])
        
        # Material properties
        material = edge_data.get('material', '')
        features.extend([
            1 if 'steel' in material.lower() else 0,
            1 if 'concrete' in material.lower() else 0,
            1 if 'aluminum' in material.lower() else 0
        ])
        
        # Section properties
        features.extend([
            float(edge_data.get('cross_section_area', 0)),
            float(edge_data.get('moment_of_inertia_y', 0)),
            float(edge_data.get('moment_of_inertia_z', 0)),
            float(edge_data.get('length', 0))
        ])
        
        return np.array(features)
    

    # VISUALISE GRAPH
    def visualize(self, figsize=(12, 10), node_size=100):
        """Visualize the structural graph in 3D"""

        if self.graph is None:
            raise ValueError("Graph not created yet. Call parse() first.")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get node positions
        pos = {node: data['coordinates'] for node, data in self.graph.nodes(data=True)}
        
        # Draw edges (structural members)
        for src, dst in self.graph.edges():
            src_pos = pos[src]
            dst_pos = pos[dst]
            ax.plot([src_pos[0], dst_pos[0]], 
                    [src_pos[1], dst_pos[1]], 
                    [src_pos[2], dst_pos[2]], 
                    'b-', linewidth=1)
        
        # Draw nodes (connection points)
        xs, ys, zs = [], [], []
        node_colors = []
        for node, data in self.graph.nodes(data=True):
            coord = data['coordinates']
            xs.append(coord[0])
            ys.append(coord[1])
            zs.append(coord[2])
            if data.get('is_support', False):
                node_colors.append('b')  # Blue for supports
            elif 'PointConnection' in data.get('ifc_type', ''):
                node_colors.append('g')  # Green for point connections
            else:
                node_colors.append('r')  # Red for other connections
        
        ax.scatter(xs, ys, zs, c=node_colors, s=node_size)
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title("Structural Graph (Nodes=Connections, Edges=Members)")
        plt.tight_layout()
        plt.show()
