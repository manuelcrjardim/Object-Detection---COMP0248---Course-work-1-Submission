import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 
import torchvision.transforms.functional as TF
from tqdm import tqdm


class HandGestureDataset(Dataset):
    def __init__(self, samples, transform=None, num_augmentations=3):
        self.data = []
        img_width, img_height = 640.0, 480.0 # Resolution fixed for all students 
        
        print(f"Loading and pre-computing dataset... (This might take a minute)")
        for sample_info in tqdm(samples, desc="Processing Samples"):
            
            # 1. Load raw data
            rgb_img = Image.open(sample_info['rgb_path']).convert('RGB')
            depth_raw = np.load(sample_info['depth_raw_path']).astype(np.float32)
            anno_img = Image.open(sample_info['anno_path']).convert('L')
            
            # 2. Convert to base tensors
            rgb_tensor = TF.to_tensor(rgb_img)
            
            depth_tensor = torch.from_numpy(depth_raw).unsqueeze(0)
            if depth_tensor.max() > 0:
                depth_tensor = depth_tensor / depth_tensor.max()
                
            mask_tensor = TF.to_tensor(anno_img)
            label = torch.tensor(sample_info['label_idx'], dtype=torch.long)
            
            # 3. Calculate absolute bounding box
            mask_np = np.array(anno_img)
            y_indices, x_indices = np.where(mask_np == 255)
            
            if len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                bbox_abs = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
            else:
                continue # Skip frames that have absolutely no mask
            
            # 4. Normalize the original box and append the ORIGINAL sample
            bbox_norm = bbox_abs / torch.tensor([img_width, img_height, img_width, img_height])
            bbox_norm = bbox_norm.squeeze()
            self.data.append((rgb_tensor, depth_tensor, label, bbox_norm, mask_tensor))
            
            # 5. The Augmentation Loop 
            if transform is not None:
                for _ in range(num_augmentations):
                    # Pass the ABSOLUTE bbox to the transform so the math works
                    aug_rgb, aug_depth, aug_mask, aug_bbox_abs = transform(
                        rgb_tensor, depth_tensor, mask_tensor, bbox_abs
                    )
                    
                    # Normalize the newly transformed box
                    aug_bbox_norm = aug_bbox_abs / torch.tensor([img_width, img_height, img_width, img_height])
                    aug_bbox_norm = aug_bbox_norm.squeeze()
                    
                    # Append the augmented sample to the dataset
                    self.data.append((aug_rgb, aug_depth, label, aug_bbox_norm, aug_mask))

        print(f"Finished! Dataset holds {len(self.data)} total tensors in memory.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]