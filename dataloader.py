import os
import random
import h5py
from sklearn.model_selection import train_test_split
import shutil
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.datapipes import functional_transform
import numpy as np
from torch_geometric.transforms import (
    Compose,
    FaceToEdge,
    Cartesian,
    GenerateMeshNormals,
    PointPairFeatures,
    BaseTransform
)

@functional_transform('compute_sdf')
class ComputeSDF(BaseTransform):
    def __init__(self, bbox_min=[0,0,0], bbox_max=[3,3,3]):
        self.bbox_min = torch.tensor(bbox_min, dtype=torch.float)
        self.bbox_max = torch.tensor(bbox_max, dtype=torch.float)

    def compute_distance_to_bbox(self, pos):
        # Ensure tensors are on same device
        self.bbox_min = self.bbox_min.to(pos.device)
        self.bbox_max = self.bbox_max.to(pos.device)
        
        # Calculate distances to each face of the box
        dx = torch.maximum(self.bbox_min[0] - pos[:,0], 
                         torch.maximum(torch.tensor(0.).to(pos.device), 
                                     pos[:,0] - self.bbox_max[0]))
        dy = torch.maximum(self.bbox_min[1] - pos[:,1], 
                         torch.maximum(torch.tensor(0.).to(pos.device), 
                                     pos[:,1] - self.bbox_max[1]))
        dz = torch.maximum(self.bbox_min[2] - pos[:,2], 
                         torch.maximum(torch.tensor(0.).to(pos.device), 
                                     pos[:,2] - self.bbox_max[2]))
        
        # Compute Euclidean distance
        distance = torch.sqrt(dx**2 + dy**2 + dz**2)
        
        # Check if points are inside box
        inside = ((pos[:,0] >= self.bbox_min[0]) & (pos[:,0] <= self.bbox_max[0]) &
                 (pos[:,1] >= self.bbox_min[1]) & (pos[:,1] <= self.bbox_max[1]) &
                 (pos[:,2] >= self.bbox_min[2]) & (pos[:,2] <= self.bbox_max[2]))
        
        # Make distance negative for inside points
        distance = torch.where(inside, -distance, distance)
        
        return distance

    def forward(self, data: Data) -> Data:
        assert 'pos' in data
        data.sdf = self.compute_distance_to_bbox(data.pos)
        return data
    
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps
                mean = self.mean[:,sample_idx]
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class SFEMInMemoryDataset(InMemoryDataset):
    def __init__(self, root, split, num_samples=None, percent_val_test=None, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.num_samples = num_samples
        self.percent_val_test = percent_val_test
        super(SFEMInMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        split_dir = os.path.join(self.root, 'raw', self.split)
        return os.listdir(split_dir)

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        
        split_dir = os.path.join(self.root, 'raw', self.split)
        files = [f for f in os.listdir(split_dir) if f.endswith('.h5')]
        
        random.shuffle(files)

        if self.split == 'train':
            num_samples = int(len(files) * self.percent_val_test)
            files = files[:num_samples]
        elif self.percent_val_test is not None and self.split in ['val', 'test']:
            num_samples = int(len(files) * self.percent_val_test)
            files = files[:num_samples]

        for file_name in tqdm(files, desc=f"Processing {self.split} data"):
            data = self.load_physics_data(split_dir, file_name)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        data_list = [self.normalize_target(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def load_physics_data(self, base_path, file_name):
        file_path = os.path.join(base_path, file_name)
        with h5py.File(file_path, 'r') as f:
            Vertices = f['Vertices'][:]
            stress_values = f['VonMises'][:]
            # stress_tensor = f['StressTensor'][:]  # Optional full stress tensor
            Cell_Data = f['Cell_Faces'][:]
            # Tetra_Data = f['Facets'][:]    # Optional volume data
            # displacement = f['u'][:]    # Optional displacement data
            load_class = f['Load_Class'][:][0].decode('utf-8')
            boundary_conditions = f['Fixed_Facet'][:]

        edge_data = torch.tensor(Cell_Data, dtype=torch.long).t().contiguous()
        x = torch.tensor(Vertices, dtype=torch.float)
        # num_vertices = x.size(0)
        y = torch.tensor(stress_values, dtype=torch.float)
        # ytensor = torch.tensor(stress_tensor, dtype=torch.float).reshape(num_vertices, -1)  # Optional full stress tensor

        # Reshape Stress_values to be a N,1 tensor
        y = y.reshape(-1, 1)
        # disp = torch.tensor(displacement, dtype=torch.float)  # Optional displacement data
        bc = torch.tensor(boundary_conditions, dtype=torch.float)

        load_class_map = {'small_Load': 0, 'medium_Load': 1, 'large_Load': 2}
        load_class_index = load_class_map[load_class]
        load_class_one_hot = torch.zeros(3, dtype=torch.float)
        load_class_one_hot[load_class_index] = 1.0
        load_class_feature = load_class_one_hot.repeat(x.size(0), 1)

        x_features = torch.cat([bc, load_class_feature], dim=1)

        data = Data(face=edge_data, pos=x, y=y, file_path=file_path)

        if self.pre_transform:
            data = self.pre_transform(data)
        
        # Change data.sdf from size N to N,1
        data.sdf = data.sdf.unsqueeze(1)

        # Add data.pos, data.norm, and data.sdf to data.x
        data.x = torch.cat([data.pos, data.sdf, data.norm, x_features], dim=1)
        
        return data

    @staticmethod
    def normalize_target(data):
        data.y_original = data.y.clone()
        # First log transform
        data.y = torch.log1p(data.y)
        # Then unit gaussian normalize
        if not hasattr(data, 'normalizer'):
            data.normalizer = UnitGaussianNormalizer(data.y)
        data.y = data.normalizer.encode(data.y)
        return data

    @staticmethod
    def unnormalize_target(data, normalized_y):
        # First undo gaussian normalization
        y = data.normalizer.decode(normalized_y)
        # Then undo log transform
        return torch.expm1(y)

def filter_zeros(data):
    threshold_value = 1E-05 
    return (data.y > threshold_value).any()

def get_transforms():
    pre_transform = Compose([
        ComputeSDF(bbox_min=[0,0,0], bbox_max=[3,3,3]), 
        FaceToEdge(remove_faces=False),
        GenerateMeshNormals(),
    ])

    transform = Compose([
        Cartesian(),
        PointPairFeatures(),
    ])
    
    return pre_transform, transform

def normalize_dataset(train_dataset, val_dataset):
    # Get max stress from both datasets
    train_max = max([data.y.max().item() for data in train_dataset])
    val_max = max([data.y.max().item() for data in val_dataset])
    max_stress = max(train_max, val_max)

    # Normalize both datasets
    for data in train_dataset:
        data.y_original = data.y.clone() 
        data.y = data.y / max_stress

    for data in val_dataset:
        data.y_original = data.y.clone()
        data.y = data.y / max_stress

    return train_dataset, val_dataset

def process_files(directory):
    """
    Process H5 files in a directory and remove any files containing NaN values.
    
    Args:
        directory (str): Path to the directory containing H5 files
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        stress_values = np.array(f['VonMises'])
                        vertices = np.array(f['Vertices'])
                        cell_data = np.array(f['Cell_Faces'])
                        
                        if np.isnan(vertices).any():
                            print(f"File {file_path} contains NaN values in vertices data. Deleting...")
                            os.remove(file_path)
                        elif np.isnan(cell_data).any():
                            print(f"File {file_path} contains NaN values in cell data. Deleting...")
                            os.remove(file_path)
                        elif np.isnan(stress_values).any():
                            print(f"File {file_path} contains NaN values in stress data. Deleting...")
                            os.remove(file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

def split_dataset_geometry(main_dir, distribution={'train': 0.85, 'val': 0.15}):
    """
    Split files in a directory into train and validation sets under a raw directory,
    ensuring geometries are not shared between sets.
    
    Args:
        main_dir (str): Path to the main directory containing the files
        distribution (dict): Dictionary specifying the split ratios (should sum to 1.0)
    """
    # Validate distribution
    if sum(distribution.values()) != 1.0:
        raise ValueError("Distribution values must sum to 1.0")
    if set(distribution.keys()) != {'train', 'val'}:
        raise ValueError("Distribution must only contain 'train' and 'val' keys")
    
    # Create raw directory
    raw_dir = os.path.join(main_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Get the list of files
    files = [f for f in os.listdir(main_dir) if f.endswith('.h5')]
    
    # Create a dictionary mapping geometries to their associated files
    geometry_files = {}
    for file in files:
        # Extract geometry identifier from the filename
        # For a file like "00204_001_final_large_Load_iter1.h5"
        # This will extract "00204_001_final"
        geometry = '_'.join(file.split('_')[:3])
        if geometry not in geometry_files:
            geometry_files[geometry] = []
        geometry_files[geometry].append(file)
    
    # Get list of unique geometries and shuffle them
    geometries = list(geometry_files.keys())
    random.shuffle(geometries)
    
    # Calculate the number of geometries for each set
    num_geometries = len(geometries)
    num_train_geometries = int(distribution['train'] * num_geometries)
    
    # Split geometries into train and val sets
    train_geometries = geometries[:num_train_geometries]
    val_geometries = geometries[num_train_geometries:]
    
    # Create subdirectories and move files
    for subdir, geom_list in zip(['train', 'val'], [train_geometries, val_geometries]):
        subdir_path = os.path.join(raw_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        
        # Move all files for each geometry in this set
        for geometry in geom_list:
            for file in geometry_files[geometry]:
                src_path = os.path.join(main_dir, file)
                dst_path = os.path.join(subdir_path, file)
                shutil.move(src_path, dst_path)
                print(f"Moved {file} to {subdir}")

def organize_simulation_results(simulation_dir="./Simulation_Results", 
                               output_dir="./dataset", 
                               train_ratio=0.85):
    """
    Automatically organize simulation H5 files into train/val splits (85/15)
    """
    # First, clean the data by removing files with NaN values
    print("Processing files to remove NaN values...")
    process_files(simulation_dir)
    
    # Use geometry-based splitting
    split_dataset_geometry(simulation_dir, {'train': train_ratio, 'val': 1-train_ratio})
    
    # Move the raw directory to the output location
    if simulation_dir != output_dir:
        raw_source = os.path.join(simulation_dir, 'raw')
        raw_dest = os.path.join(output_dir, 'raw')
        if os.path.exists(raw_dest):
            shutil.rmtree(raw_dest)
        shutil.move(raw_source, raw_dest)
    
    # Count final files
    train_files = len([f for f in os.listdir(f"{output_dir}/raw/train") if f.endswith('.h5')])
    val_files = len([f for f in os.listdir(f"{output_dir}/raw/val") if f.endswith('.h5')])
    total_files = train_files + val_files
    
    print(f"Organized {total_files} files:")
    print(f"  Train: {train_files} files ({train_files/total_files*100:.1f}%)")
    print(f"  Val: {val_files} files ({val_files/total_files*100:.1f}%)")
    
    return output_dir

def create_datasets(simulation_dir="./Simulation_Results", 
                   root_dir=None, 
                   auto_organize=True,
                   percent_val_test=1):
    
    if auto_organize:
        if root_dir is None:
            root_dir = "./dataset"
        root_dir = organize_simulation_results(simulation_dir, root_dir)
        print(f"Files organized in: {root_dir}")
    
    pre_transform, transform = get_transforms()

    train_dataset = SFEMInMemoryDataset(
        root=root_dir,
        split='train', 
        percent_val_test=percent_val_test,
        pre_transform=pre_transform,
        transform=transform,
        pre_filter=filter_zeros
    )

    val_dataset = SFEMInMemoryDataset(
        root=root_dir,
        split='val',
        percent_val_test=percent_val_test, 
        pre_transform=pre_transform,
        transform=transform,
        pre_filter=filter_zeros
    )
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = create_datasets()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")