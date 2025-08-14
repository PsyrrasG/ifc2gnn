# IFC Structural Graph Parser

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Convert IFC structural models to network graphs for GNN-based applications with PyTorch.

Note: This is a beta version. Please report any issues you encounter.

## Features
- Detection of nodes (joints/supports) & elements (beams/columns)
- Structure automatically converted to network graph
- Generation of node feature matrix, edge features, and adjacency matrix. 
- ML Ready: Export to PyTorch Geometric Data object
- 3D graph visualization with Matplotlib
- Built-in topology verification
- Works with both IFC4 and IFC2x3 Structural Analysis View files

## Installation
```bash
git clone https://github.com/PsyrrasG/ifc2gnn.git
cd ifc2gnn
pip install -e .
