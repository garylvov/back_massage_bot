import hydra
import torch

import sys
import os
sys.path.append('/back_massage_bot/external/Human3D/')

from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(
        self,
        x,
        point2segment=None,
        raw_coordinates=None,
        is_eval=True,
        clip_feat=None,
        clip_pos=None,
    ):
        x = self.model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
            clip_feat=clip_feat,
            clip_pos=clip_pos,
        )
        return x
    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from hydra.experimental import initialize_config_module

# imports for input loading
import albumentations as A
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d

def get_model(checkpoint_path=None):
    # Store original working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the directory where the model code is located
        human3d_dir = '/back_massage_bot/external/Human3D'
        os.chdir(human3d_dir)
        
        # Print directory structure for debugging
        print(f"Current directory: {os.getcwd()}")
        
        # List contents to verify conf directory exists
        print(f"Directory contents: {os.listdir('.')}")
        
        # Check if conf directory exists
        if not os.path.exists('conf'):
            print(f"ERROR: 'conf' directory not found in {os.getcwd()}")
            raise FileNotFoundError(f"Config directory not found at {os.path.join(os.getcwd(), 'conf')}")
        
        # Reset Hydra's global state
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        # Initialize Hydra directly from conf directory
        # Hydra is run from the current directory
        os.environ["HYDRA_CONFIG_PATH"] = os.path.join(os.getcwd(), "conf")
        initialize_config_module(config_module="conf")
        
        # Compose the configuration
        cfg = compose(
            config_name="config_base_instance_segmentation", 
            overrides=[
                f"general.checkpoint={checkpoint_path}",
                "general.train_mode=False",
                "model=mask3d_hp",
                "general.body_part_segmentation=True",
                "data.batch_size=1",
                "data.num_workers=0",
                "data/datasets=egobody"
            ]
        )
        
        # Instantiate the model using the loaded configuration
        model = InstanceSegmentation(cfg)
        
        # Load checkpoint
        if checkpoint_path is not None:
            cfg.general.checkpoint = checkpoint_path
            # Set backbone_checkpoint to the same value as checkpoint
            cfg.general.backbone_checkpoint = checkpoint_path
            cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
            
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)
    
    return model


def load_mesh(pcl_file):
    # Check file extension to determine how to load
    if pcl_file.endswith('.ply') or pcl_file.endswith('.obj'):
        try:
            # Try to load as a mesh first
            mesh = o3d.io.read_triangle_mesh(pcl_file)
            
            # If no triangles were loaded, it might be a point cloud
            if len(mesh.triangles) == 0:
                print("File appears to be a point cloud rather than a mesh, loading as point cloud")
                pcd = o3d.io.read_point_cloud(pcl_file)
                # Convert point cloud to mesh (just vertices, no triangles)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = pcd.points
                if len(pcd.colors) > 0:
                    mesh.vertex_colors = pcd.colors
            
            points = np.asarray(mesh.vertices)
            if len(mesh.vertex_colors) == 0:
                print("No vertex colors found, using default white color")
                colors = np.ones((len(points), 3))
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                
            return mesh
            
        except Exception as e:
            print(f"Error loading mesh: {e}")
            raise
    else:
        raise ValueError(f"Unsupported file format: {pcl_file}")

def prepare_data(mesh, device):
    
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    
    points = np.asarray(mesh.vertices)
    if len(mesh.vertex_colors) == 0:
        # Default color - white
        colors = np.full((len(points), 3), 255, dtype=np.uint8)
    else:
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    
    # fix rotation bug
    # points = points[:, [0, 2, 1]]
    # points[:, 2] = -points[:, 2]

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=torch.from_numpy(coords).contiguous(),
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    
    
    return data, points, colors, features, unique_map, inverse_map


def map_output_to_pointcloud(mesh, 
                             outputs, 
                             inverse_map, 
                             label_space='scannet200',
                             confidence_threshold=0.9):
    
    # parse predictions
    # Check which keys are available in the output
    print("Available keys in output:", outputs.keys())
    
    # Use pred_part_logits instead of pred_human_logits if that's what's available
    logits_key = "pred_part_logits" if "pred_part_logits" in outputs else "pred_logits"
    logits = outputs[logits_key]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()

    labels = []
    confidences = []
    masks_binary = []

    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200 and c > confidence_threshold:
            labels.append(l.item())
            confidences.append(c.item())
            masks_binary.append(
                m[inverse_map])  # mapping the mask back to the original point cloud
    
    # save labelled mesh
    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles

    labels_mapped = np.zeros((len(mesh.vertices), 1))

    for i, (l, c, m) in enumerate(
        sorted(zip(labels, confidences, masks_binary), reverse=False)):
        
        if label_space == 'scannet200':
            label_offset = 1
            
            l = int(l) + label_offset
                        
        labels_mapped[m == 1] = l
        
    return labels_mapped


def save_colorized_mesh(mesh, labels_mapped, output_file):
    # Define a simple color map for two classes: 0 (background) and 1 (human)
    color_map = {
        0: [255, 255, 255],  # White for background
        1: [255, 0, 0]       # Red for human
    }
    
    # Initialize a color array for all vertices in the mesh
    colors = np.zeros((len(mesh.vertices), 3))
    
    # Get unique labels within the mapped labels
    unique_labels = np.unique(labels_mapped)
    print(unique_labels)
    
    # Apply colors based on the unique labels found in labels_mapped
    for li in unique_labels:
        if li in color_map:
            # Apply color to vertices where label matches
            colors[(labels_mapped == li)[:, 0], :] = color_map[li]
        else:
            # Handle unexpected label
            raise ValueError(f"Label {li} not supported by the defined color map.")
    
    # Normalize the color values to be between 0 and 1
    colors = colors / 255.0
    
    # Assign colors to mesh vertices
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Write the colorized mesh to the specified output file
    o3d.io.write_triangle_mesh(output_file, mesh)



if __name__ == '__main__':
    
    model = get_model('/models/human3d.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # load input data
    pointcloud_file = 'test.ply'
    print(f"Loading point cloud from: {pointcloud_file}")
    mesh = load_mesh(pointcloud_file)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices")
    
    # prepare data
    print("Preparing data for inference...")
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)
    
    # run model
    print("Running inference...")
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
    
    # Print all output keys and shapes for debugging
    print("\nModel output details:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            print(f"{key}: list of tensors with shape={value[0].shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # map output to point cloud
    print("\nMapping output to point cloud...")
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map, confidence_threshold=0.3)
    
    # save colorized mesh
    output_file = 'mesh.ply'
    print(f"Saving colorized mesh to: {output_file}")
    save_colorized_mesh(mesh, labels, output_file)
    print("Done!")