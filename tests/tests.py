import sys
sys.path.append("../ifc2gnn")

from ifc2gnn.ifc2gnn import IFCStructuralGraphParser

def test_graph_creation(sample_ifc):
    """Test basic graph parsing"""
    parser = IFCStructuralGraphParser(sample_ifc)
    graph = parser.parse()
    assert len(graph.nodes) > 0, "No nodes were created"
    assert len(graph.edges) > 0, "No edges were created"
    print("Test successful.")


def test_graph_matches_structure(sample_ifc):
    """
    Verify that:
    - Graph nodes = Number of joints/supports in IFC
    - Graph edges = Number of structural members/elements in IFC
    """
    import ifcopenshell
    # Load IFC file directly for validation
    ifc_file = ifcopenshell.open(sample_ifc)
    # Count structural elements in IFC
    joints = (
        len(ifc_file.by_type("IfcStructuralPointConnection")) +
        len(ifc_file.by_type("IfcStructuralCurveConnection")) +
        len(ifc_file.by_type("IfcStructuralPointReaction"))
    )
    structural_members = (
        len(ifc_file.by_type("IfcBeam")) +
        len(ifc_file.by_type("IfcColumn")) +
        len(ifc_file.by_type("IfcStructuralMember"))
    )
    # Parse with our tool
    parser = IFCStructuralGraphParser(sample_ifc)
    graph = parser.parse()
    # Assertions
    assert len(graph.nodes) == joints, (
        f"Node count ({len(graph.nodes)}) should equal "
        f"joints+supports count ({joints})"
    )
    assert len(graph.edges) == structural_members, (
        f"Edge count ({len(graph.edges)}) should equal "
        f"structural members count ({structural_members})"
    )
    print("Test successful.")


def test_visualization(sample_ifc):
    """Smoke test for visualization"""
    parser = IFCStructuralGraphParser(sample_ifc)
    graph = parser.parse()
    parser.visualize()  # Just verify no errors
    print("Test successful.")