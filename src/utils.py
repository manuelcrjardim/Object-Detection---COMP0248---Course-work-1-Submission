import torch
from pathlib import Path
from torchvision.transforms import v2
from torchvision import tv_tensors

def collate_fn_filter_none(batch):
    ''' Function  used to collate batch'''
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

# Class defininf the different augmentations performed on the dataset
class MultiTaskAugmentation:
    def __init__(self, is_train=True):
        self.is_train = is_train
        
        # color transform
        self.color_transforms = v2.Compose([
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        ])
        
        # spatial transform
        self.spatial_transforms = v2.Compose([
            v2.RandomRotation(degrees=15),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

    def __call__(self, rgb, depth, mask, bbox_coords):
        if not self.is_train:
            return rgb, depth, mask, bbox_coords

        rgb_tv = tv_tensors.Image(rgb) 
        depth_tv = tv_tensors.Image(depth)
        mask_tv = tv_tensors.Mask(mask)
        
        bbox_tv = tv_tensors.BoundingBoxes(
            bbox_coords, 
            format=tv_tensors.BoundingBoxFormat.XYXY, 
            canvas_size=(480, 640)
        )

        rgb_tv = self.color_transforms(rgb_tv)

        rgb_tv, depth_tv, mask_tv, bbox_tv = self.spatial_transforms(
            rgb_tv, depth_tv, mask_tv, bbox_tv
        )

        return (
            rgb_tv.as_subclass(torch.Tensor), 
            depth_tv.as_subclass(torch.Tensor), 
            mask_tv.as_subclass(torch.Tensor), 
            bbox_tv.as_subclass(torch.Tensor)
        )
    
def calc_bbox_iou(pred_box, target_box):
    """Calculates the Intersection over Union (IoU) for bounding boxes."""
    x1 = torch.max(pred_box[:, 0], target_box[:, 0])
    y1 = torch.max(pred_box[:, 1], target_box[:, 1])
    x2 = torch.min(pred_box[:, 2], target_box[:, 2])
    y2 = torch.min(pred_box[:, 3], target_box[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = torch.clamp(pred_box[:, 2] - pred_box[:, 0], min=0) * torch.clamp(pred_box[:, 3] - pred_box[:, 1], min=0)
    target_area = torch.clamp(target_box[:, 2] - target_box[:, 0], min=0) * torch.clamp(target_box[:, 3] - target_box[:, 1], min=0)

    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-6) 
    return iou

def calc_mask_metrics(pred_logits, target_mask):
    """Calculates Mask IoU and Dice Coefficient."""
    pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
    target_mask = target_mask.float()
    
    intersection = (pred_mask * target_mask).sum(dim=(1, 2, 3))
    pred_sum = pred_mask.sum(dim=(1, 2, 3))
    target_sum = target_mask.sum(dim=(1, 2, 3))
    
    union = pred_sum + target_sum - intersection
    
    iou = intersection / (union + 1e-6)
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-6)
    
    return iou, dice

def gather_test_samples(data_path_str):
    """
    Function used to collect the samples from the test set 
    """
    data_path = Path(data_path_str)
    all_samples = []

    # Defining the order in which the classes will be indexed is essential to ensure that the it matches the order during training
    classes = ['G07_peace', 'G10_three', 'G08_rock', 'G04_ok', 'G03_like', 'G09_stop', 'G01_call', 'G06_palm', 'G02_dislike', 'G05_one']

    print(f"Parsing test dataset directory: {data_path_str}...")

    # Looping through test set, expects the test set to contain the gesture directories inside it directly
    for gesture_dir in data_path.iterdir():
        if not gesture_dir.is_dir(): continue  

        gesture_label = gesture_dir.name
        
        if gesture_label not in classes:
            continue
            
        class_idx = classes.index(gesture_label)

        for clip_dir in sorted(gesture_dir.iterdir()):
            if not clip_dir.is_dir(): continue
            
            rgb_dir = clip_dir / 'rgb'
            if not rgb_dir.exists(): continue
                
            for rgb_file in sorted(rgb_dir.glob('frame_*.png')):
                frame_name = rgb_file.stem
                
                depth_file = clip_dir / 'depth' / f'{frame_name}.png'
                depth_raw_file = clip_dir / 'depth_raw' / f'{frame_name}.npy'
                anno_dir = clip_dir / 'annotation'
                
                anno_file_strict = anno_dir / f'{frame_name}.png'
                anno_file_messy = anno_dir / f'{gesture_label}_{clip_dir.name}_rgb_{frame_name}.png'
                
                final_anno_path = None
                if anno_file_strict.exists():
                    final_anno_path = anno_file_strict
                elif anno_file_messy.exists():
                    final_anno_path = anno_file_messy
                    
                if final_anno_path is not None:
                    all_samples.append({
                        'rgb_path': rgb_file,
                        'depth_raw_path': depth_raw_file,
                        'anno_path': final_anno_path,
                        'label_idx': class_idx,
                        'depth_path': depth_file,
                    })
                            
    print(f"Found {len(all_samples)} valid test samples across {len(classes)} classes.")
    return all_samples, len(classes)