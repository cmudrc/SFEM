# SFEM Dataset - Neural Network Surrogate Modeling

This repository provides a dataset for training neural network surrogate models for the stochastic finite element method (SFEM). The dataset contains 3D finite element meshes with corresponding stress distributions under stochastic point elastic loading conditions, designed for evaluating graph neural network architectures for engineering design applications.

## Overview

Modern engineering design requires probabilistic simulation to account for uncertainties in geometry and loading conditions. Traditional SFEM uses thousands of deterministic FEM evaluations, creating prohibitive computational costs. This dataset enables training neural network surrogates that can predict stress distributions orders of magnitude faster than traditional SFEM (milliseconds versus hours), supporting efficient iterative design exploration.

## Installation

### FEM Simulation Environment
For running finite element simulations and generating H5 data files:

```bash
conda create -n SFEM
conda activate SFEM
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install h5py gmsh meshio
```

FEniCSx is the latest iteration of FEniCS and is the recommended finite element library. Installation instructions and additional options can be found at https://fenicsproject.org/download/

### Machine Learning Environment  
For working with existing H5 simulation results:

```bash
conda create -n SFEMGraph
conda activate SFEMGraph
pip install torch torchvision
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
pip install torch-geometric h5py scikit-learn tqdm
```

Tested on Python 3.12.7, CUDA 12.6. **Note**: PyTorch Geometric requires exact version matching. Check your versions:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

## Usage

### Data Generation
1. Generate meshes: `python mesh_generation.py`
2. Run simulations: `python simulation.py`
3. Load datasets: `python dataloader.py`

### Dataset Structure
- **Input**: 3D finite element meshes
- **Output**: Von Mises stress distributions
- **Features**: Vertex positions, signed distance function, mesh normals, boundary conditions, load classes
- **Split**: 85% train, 15% validation (geometry-based splitting)

## Dataset Access

The complete dataset used in this work is available on HuggingFace:

[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/cmudrc/SFEM)

### Dataset Statistics
- **Total size**: ~23GB
- **STEP files**: 39,894 geometries (3GB) - Input for `mesh_generation.py`
- **H5 simulation results**: 48,803 files - Output of `dataloader.py`
  - Training: 7GB
  - Validation: 1.2GB
  - Processed PyTorch data: 12GB - Ready for any PyTorch Geometric workflow

## Dataset Details

### Input Geometries
The STEP files used in this dataset were generated using [BrepGen](https://github.com/samxuxiang/BrepGen), containing approximately 16,000 CAD geometries. These provide diverse 3D shapes for comprehensive finite element analysis.

### Simulation Results
Each simulation result contains:
- Vertices: 3D coordinates (N × 3)
- VonMises: Stress values (N × 1) under stochastic point elastic loading
- Load classes: Small (200N), medium (2000N), large (20000N) point loads
- Boundary conditions: Fixed/free vertex masks
- Material: E = 2.303 GPa, ν = 0.4002

## Citation

@dataset{ezemba2025sfem,
  title={SFEM Dataset - Neural Network Surrogate Modeling for Stochastic FEM using 3D Graph Representations},
  author={Ezemba, Jessica},
  year={2025},
  publisher={Journal of Mechanical Design},
}

