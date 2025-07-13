"""
Author: Jessica Ezemba
Date: 2025-07-13
Description: Given .msh files, this script processes them for stochastic finite element simulations.
License: MIT
"""

import numpy as np
import meshio
from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl
import h5py
import os
from functools import lru_cache
import json
import time
import gc

def load_progress(progress_file):
    """Load the list of processed files from progress tracker"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'processed': [], 'skipped': [], 'failed': []}

def update_progress(progress_file, file_name, status):
    """Update the progress tracker with a new file"""
    progress = load_progress(progress_file)
    progress[status].append(file_name)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=4)

def get_node_count(mesh_file_path):
    """Get the number of nodes in the mesh"""
    mesh_io_file = meshio.read(mesh_file_path)
    return len(mesh_io_file.points)

def generate_random_clumps(free_facets, num_clumps, min_clump_size, max_clump_size):
    """Generate random clumps of facets for impact zones"""
    # Pre-calculate random sizes for all clumps at once
    clump_sizes = np.random.randint(min_clump_size, max_clump_size + 1, num_clumps)
    # Pre-calculate start indices for all clumps at once
    max_start_indices = len(free_facets) - clump_sizes + 1
    start_indices = np.random.randint(0, max_start_indices)
    
    # Create clumps
    return [free_facets[start:start + size] 
            for start, size in zip(start_indices, clump_sizes)]

@lru_cache(maxsize=128)
def create_mesh(mesh_file_path, cell_type):
    """Create mesh from meshio file with caching"""
    mesh_io_file = meshio.read(mesh_file_path)
    cells = mesh_io_file.get_cells_type(cell_type)
    cell_data = mesh_io_file.get_cell_data("gmsh:physical", cell_type)
    
    out_mesh = meshio.Mesh(
        points=mesh_io_file.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh

def simulate_impact(mesh_file_path, output_dir, progress_file, max_nodes=100000, num_iterations=50):
    """Main simulation function with node count check and progress tracking"""
    try:
        # Check if file has already been processed
        progress = load_progress(progress_file)
        base_name = os.path.basename(mesh_file_path)
        
        if base_name in progress['processed']:
            print(f"Skipping {base_name} - already processed")
            return True
        if base_name in progress['skipped']:
            print(f"Skipping {base_name} - previously skipped due to node count")
            return True
        if base_name in progress['failed']:
            print(f"Retrying {base_name} - previously failed")
            
        # Check node count
        node_count = get_node_count(mesh_file_path)
        print(f"Node count for {base_name}: {node_count:,}")
        
        if node_count > max_nodes:
            print(f"Skipping {base_name} - node count ({node_count:,}) exceeds maximum ({max_nodes:,})")
            update_progress(progress_file, base_name, 'skipped')
            return True

        print(f"Processing file {base_name}")
        start_time = time.time()
        
        # Material parameters
        E = 2.303e9
        nu = 0.4002
        mu = E/(2.0*(1.0 + nu))
        lambda_ = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
        
        load_classes = {
            'small_Load': {'force': 200, 'num_clumps': 5, 'min_size': 3, 'max_size': 10},
            'medium_Load': {'force': 2000, 'num_clumps': 10, 'min_size': 6, 'max_size': 20},
            'large_Load': {'force': 20000, 'num_clumps': 20, 'min_size': 10, 'max_size': 30},
        }
        
        # Read mesh file once
        mesh_io_file = meshio.read(mesh_file_path)
        base_name = os.path.splitext(os.path.basename(mesh_file_path))[0]
        
        # Optimize XDMF file creation
        if MPI.COMM_WORLD.rank == 0:
            tetra_mesh = create_mesh(mesh_file_path, "tetra")
            triangle_mesh = create_mesh(mesh_file_path, "triangle")
            meshio.write("temp_mesh_tetra.xdmf", tetra_mesh)
            meshio.write("temp_mesh_triangle.xdmf", triangle_mesh)
        MPI.COMM_WORLD.barrier()
        
        # Optimize mesh reading
        with XDMFFile(MPI.COMM_WORLD, "temp_mesh_tetra.xdmf", "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            ft = xdmf.read_meshtags(domain, name="Grid")
        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
        with XDMFFile(MPI.COMM_WORLD, "temp_mesh_triangle.xdmf", "r") as xdmf:
            ct = xdmf.read_meshtags(domain, name="Grid")
        
        # Pre-compute mesh topology
        tdim = domain.topology.dim
        fdim = tdim - 1
        # boundary_facets = mesh.exterior_facet_indices(domain.topology)
        
        # Pre-compute topology data
        domain.topology.create_connectivity(2, 3)
        domain.topology.create_connectivity(2, 0)
        domain.topology.create_connectivity(3, 0)
        
        # Optimize cell data processing
        cell_data = mesh_io_file.get_cell_data("gmsh:physical", "triangle")
        unique_tags, counts = np.unique(cell_data, return_counts=True)
        most_frequent_group = unique_tags[np.argmax(counts)]

        # Get Fixed Surface
        fixed_facet = ct.find(most_frequent_group)

        # Get Free Surfaces
        free = np.setdiff1d(unique_tags, most_frequent_group).astype(np.int32)
        free_facets = np.concatenate([ct.find(tag) for tag in free])
        
        # Pre-compute function spaces
        V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
        V_scalar = fem.functionspace(domain, ("Lagrange", 1))
        V_tensor = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,domain.geometry.dim)))

        # Get size
        num_vertices = domain.geometry.x.shape[0]
        
        cell_to_vertex = domain.topology.connectivity(2, 0)
        facet_to_vertex = domain.topology.connectivity(3, 0)

        # Get cell data
        Cell_Data = np.vstack([cell_to_vertex.links(i) for i in range(domain.topology.index_map(2).size_local)])
        Facets_Data = np.vstack([facet_to_vertex.links(i) for i in range(domain.topology.index_map(3).size_local)])

        # Initialize combined binary array
        fixed_free_binary = np.zeros((num_vertices, 2), dtype=int)

        # Set fixed and free facets directly
        fixed_free_binary[np.vstack([cell_to_vertex.links(i) for i in fixed_facet]), 0] = 1
        fixed_free_binary[np.vstack([cell_to_vertex.links(i) for i in np.sort(free_facets)]), 1] = 1
        
        # Pre-define strain and stress
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        def sigma(u):
            return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
        
        # Pre-compute problem components
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        force = fem.Constant(domain, default_scalar_type((0, 0, 0)))
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        
        # Setup boundary conditions
        u_D = np.array([0, 0, 0], dtype=default_scalar_type)
        bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, fixed_facet), V)
        
        for load_class, params in load_classes.items():
            # Pre-allocate arrays
            displacement_data = np.zeros((num_iterations, num_vertices * 3))  # 3 components per vertex
            stress_data = np.zeros((num_iterations, num_vertices))  # 1 component per vertex
            stress_tensor_data = np.zeros((num_iterations, num_vertices * 9))  # 9 components per vertex
            
            for iter in range(num_iterations):
                if iter % 5 == 0:
                    print(f"Processing iteration {iter + 1}/{num_iterations} for {load_class}")
                
                impact_zones = generate_random_clumps(
                    free_facets, 
                    params['num_clumps'],
                    params['min_size'],
                    params['max_size']
                )
                
                for impact_zone in impact_zones:
                    boundary_facets_tag = mesh.meshtags(
                        domain, fdim, impact_zone,
                        np.full_like(impact_zone, 1).astype(np.int32)
                    )
                    ds = ufl.Measure("ds", domain=domain, subdomain_data=boundary_facets_tag, subdomain_id=1)
                    
                    L = ufl.dot(force, v) * ufl.dx + ufl.inner(-params['force'] * ufl.FacetNormal(domain), v) * ds
                    
                    # Solve using LU solver
                    problem = LinearProblem(a, L, bcs=[bc], 
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
                    uh = problem.solve()
                    
                    s = sigma(uh) - 1./3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
                    von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
                    
                    stress_expr = fem.Expression(von_Mises, V_scalar.element.interpolation_points())
                    stresses = fem.Function(V_scalar)
                    stresses.interpolate(stress_expr)

                    # Interpolate full stress tensor
                    stress_tensor_expr = fem.Expression(s, V_tensor.element.interpolation_points())
                    stress_tensor = fem.Function(V_tensor)
                    stress_tensor.interpolate(stress_tensor_expr)
                    
                    # Store results directly in pre-allocated arrays
                    displacement_data[iter] = uh.x.array
                    stress_data[iter] = stresses.x.array
                    stress_tensor_data[iter] = stress_tensor.x.array

                    # Clean up to free memory
                    del uh, stresses, stress_tensor
                    gc.collect()
            
            # Calculate medians and reshape
            median_displacement = np.median(displacement_data, axis=0).reshape((num_vertices, 3))
            median_stress = np.median(stress_data, axis=0).reshape((num_vertices, 1))
            median_stress_tensor = np.median(stress_tensor_data, axis=0).reshape((num_vertices, 3, 3))
            
            # Batch file writing
            h5_file_name = f"{base_name}_{load_class}.h5"
            h5_file_path = os.path.join(output_dir, h5_file_name)
            
            with h5py.File(h5_file_path, 'w') as f:
                f.create_dataset('Vertices', data=domain.geometry.x)
                f.create_dataset('u', data=median_displacement)
                f.create_dataset('VonMises', data=median_stress)
                f.create_dataset('StressTensor', data=median_stress_tensor)
                f.create_dataset('Cell_Faces', data=Cell_Data)
                f.create_dataset('Facets', data=Facets_Data)
                f.create_dataset('Load_Class', data=np.array([load_class], dtype='S'))
                f.create_dataset('Fixed_Facet', data=fixed_free_binary)
            
            print(f"Saved results to {h5_file_path}")
        
        # Update progress after successful completion
        update_progress(progress_file, base_name, 'processed')
        elapsed_time = time.time() - start_time
        print(f"Completed {base_name} in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error processing file {mesh_file_path}: {e}")
        update_progress(progress_file, os.path.basename(mesh_file_path), 'failed')
        return False

if __name__ == "__main__":
    mesh_dir = "./Mesh_Files"
    output_dir = "./Simulation_Results"
    progress_file = os.path.join(output_dir, "processing_progress.json")
    max_nodes = 2000  # Set your desired maximum node count here
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load existing progress
    progress = load_progress(progress_file)
    
    # Get list of all mesh files
    files = [f for f in os.listdir(mesh_dir) if f.endswith('.msh')]
    
    # Filter out already processed files (removing .msh extension for comparison)
    remaining_files = [f for f in files 
                      if os.path.splitext(f)[0] not in progress['processed']]
    total_files = len(remaining_files)
    
    print(f"\nStarting processing of remaining {total_files} files")
    print(f"Previously processed: {len(progress['processed'])}")
    print(f"Previously skipped: {len(progress['skipped'])}")
    print(f"Previously failed: {len(progress['failed'])}")
    print("\nFiles already processed:")
    for f in progress['processed']:
        print(f"- {f}")
    print("\nProcessing remaining files:")
    
    # Process only remaining files
    for i, mesh_file in enumerate(remaining_files, 1):
        mesh_path = os.path.join(mesh_dir, mesh_file)
        base_name = os.path.splitext(mesh_file)[0]  # Remove .msh extension
        print(f"\nProcessing file {i}/{total_files}: {mesh_file}")
        simulate_impact(mesh_path, output_dir, progress_file, max_nodes, num_iterations=50)
        
    # Print final summary
    final_progress = load_progress(progress_file)
    print("\nProcessing complete!")
    print(f"Successfully processed: {len(final_progress['processed'])}")
    print(f"Skipped due to node count: {len(final_progress['skipped'])}")
    print(f"Failed: {len(final_progress['failed'])}")
    
    if final_progress['failed']:
        print("\nFailed files:")
        for failed_file in final_progress['failed']:
            print(f"- {failed_file}")