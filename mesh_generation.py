"""
Author: Jessica Ezemba
Date: 2025-07-13
Description: Given step files, this script generates .msh files using Gmsh.
License: MIT
"""

import gmsh
import sys
import os
import subprocess
from multiprocessing import Process, Queue
import time

def process_single_file(step_file, output_dir, lc, queue):
    """
    Process a single STEP file in a separate process
    """
    try:
        # Initialize Gmsh
        gmsh.initialize()

        # Do not print messages to the terminal
        gmsh.option.setNumber("General.Terminal", 0)

        # Set global mesh options
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.option.setNumber("Mesh.Algorithm", 6)    # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        # Merge the STEP file into Gmsh
        gmsh.merge(step_file)
        gmsh.model.mesh.createGeometry()
        gmsh.model.occ.synchronize()

        # Remove all duplicates
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # Assign physical groups for volumes
        volumes = gmsh.model.occ.getEntities(dim=3)
        if not volumes:
            queue.put(("warning", f"No volumes found in '{step_file}'. Skipping this file."))
            gmsh.finalize()
            return

        volume_tags = [vol[1] for vol in volumes]
        gmsh.model.addPhysicalGroup(3, volume_tags, tag=1)

        # Assign physical groups for surfaces
        surfaces = gmsh.model.occ.getEntities(dim=2)
        surface_tags = [s[1] for s in surfaces]
        for surface_tag in surface_tags:
            gmsh.model.addPhysicalGroup(2, [surface_tag], tag=surface_tag)

        gmsh.model.occ.synchronize()

        # Mesh generation
        gmsh.model.mesh.generate(3)
        gmsh.model.occ.synchronize()

        # Create the output filename
        base_name = os.path.basename(step_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, name_without_ext + '.msh')

        # Write the mesh to file
        gmsh.write(output_file)
        queue.put(("success", f"Mesh written to '{output_file}'."))

    except Exception as e:
        queue.put(("error", f"Error processing '{step_file}': {e}"))
    finally:
        gmsh.finalize()

def main():
    # Base directory to search for STEP files
    base_path = './Step_Files/'
    
    # Output directory to save the generated .msh files
    output_dir = './Mesh_Files/'
    os.makedirs(output_dir, exist_ok=True)

    # Characteristic length for meshing
    lc = 0.1

    # Timeout in seconds (60 seconds = 1 minute)
    TIMEOUT = 60

    # Find all STEP files
    step_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.step'):
                step_file_path = os.path.join(root, file)
                step_files.append(step_file_path)

    if not step_files:
        print(f"No STEP files found in '{base_path}'.")
        sys.exit(1)

    # Process each STEP file with timeout
    for step_file in step_files:
        print(f"\nProcessing '{step_file}'...")
        
        # Create a queue for process communication
        queue = Queue()
        
        # Create and start the process
        process = Process(target=process_single_file, args=(step_file, output_dir, lc, queue))
        start_time = time.time()
        process.start()
        
        # Wait for the process to complete or timeout
        process.join(TIMEOUT)
        
        if process.is_alive():
            # If process is still running after timeout, terminate it
            process.terminate()
            process.join()
            print(f"Timeout: Processing of '{step_file}' took longer than {TIMEOUT} seconds. Skipping file.")
            continue
        
        # Get the result from the queue if available
        while not queue.empty():
            msg_type, message = queue.get()
            if msg_type == "error":
                print(f"Error: {message}")
            elif msg_type == "warning":
                print(f"Warning: {message}")
            else:
                print(message)

    print("\nProcessing completed.")

if __name__ == "__main__":
    main()