import argparse
import yaml
import torch
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader

from Backbone.pointfusion import Fusion
from DataProcess.kitti_dataset import KittiDataset
from Utils.visualization import show_image_with_boxes, display_lidar, visualize_result

def inference(args):
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Fusion(backbone=cfg['dataset'].get('backbone', 'resnet')).to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}. Using random weights.")

    model.eval()

    # Load dataset
    dataset = KittiDataset(root=cfg['dataset']['root_dir'], mode='test')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    print(f"Starting inference on {len(dataset)} samples...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_samples:
                break
            
            img, points = batch['roi_img'], batch['roi_pc']
            sample_id = batch['sample_id'][0] if 'sample_id' in batch else i
            
            # Prepare input
            img_tensor = torch.from_numpy(img).float().to(device)
            points_tensor = torch.from_numpy(points).float().to(device)
            
            # Forward pass
            pred_offset, scores = model(img_tensor, points_tensor)
            
            # Process results
            # Finding anchor point and predicted offset based on maximum score
            # Since batch size is 1, we take the first element
            max_inds = scores.max(dim=1)[1].cpu().numpy()
            
            # Assuming batch size of 1 for simplicity in visualization loop
            # But model output is [B, D, 8, 3]
            # Here we iterate over the batch items (which is just 1)
            
            # Visualization logic similar to train.py but for inference
            # We need to reconstruct the boxes
            
            # Let's visualize the first object in the batch
            idx = 0 
            max_ind = max_inds[idx]
            
            pred_off = pred_offset[idx][max_ind].cpu().numpy()
            anchor = points[idx][max_ind]
            
            # Reconstruct box
            pred_box = pred_off + anchor
            
            print(f"Sample {sample_id}: Predicted box center around {np.mean(pred_box, axis=0)}")
            
            if args.visualize:
                # This is a simplified visualization. 
                # For full visualization we might need more context like calibration which is in the dataset
                # but not directly passed here easily without modifying dataset to return it.
                # However, KittiDataset returns 'images' in sample_info if we look at __getitem__
                # But collate_fn might structure it differently.
                
                # Let's try to use the Utils.visualization functions if possible
                # visualize_result(np.array([anchor]), np.array([pred_off]), np.array([pred_box])) # Dummy gt
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for PointFusion")
    parser.add_argument("--config", type=str, default="Config/train_test.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    
    args = parser.parse_args()
    inference(args)
