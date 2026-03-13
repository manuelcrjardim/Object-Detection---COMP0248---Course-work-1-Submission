import torch
import random
import gc
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.nn as nn
from tqdm import tqdm
import json

from model import MultiTaskRGBDNet
from dataloader import HandGestureDataset
from utils import MultiTaskAugmentation
from evaluate import evaluate_model

def train_model(model, train_loader, optimizer, num_epochs=100, epoch_margin=10, device='cuda:3'):
    model.train()
    
    # individual reward functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    criterion_mask = nn.BCEWithLogitsLoss()
    
    # respective reward function weights
    w_cls = 1.0
    w_box = 15.0  
    w_mask = 2.0  
    
    loss_history = []
    lowest_loss = None
    epochs_since_lowest_loss = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            rgb, depth, target_cls, target_box, target_mask = [tensor.to(device) for tensor in batch]

            optimizer.zero_grad()
            
            pred_cls, pred_box, pred_mask = model(rgb, depth)
            
            loss_cls = criterion_class(pred_cls, target_cls)
            loss_box = criterion_bbox(pred_box, target_box)
            loss_mask = criterion_mask(pred_mask, target_mask) 
            
            # Weighted composite loss function
            total_loss = (w_cls * loss_cls) + (w_box * loss_box) + (w_mask * loss_mask)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            pbar.set_postfix({
                'Tot': f"{total_loss.item():.2f}", 
                'Cls': f"{loss_cls.item():.2f}", 
                'Box': f"{loss_box.item():.4f}", 
                'Msk': f"{loss_mask.item():.2f}"
            })
            
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # Early stopping logic 
        if lowest_loss is None or epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            epochs_since_lowest_loss = 0
            print(f'./ New lowest loss, of {lowest_loss}')
        else:
            print(f'It has been {epochs_since_lowest_loss} epochs since the lowest loss was achieved')
            epochs_since_lowest_loss += 1
            
        if epochs_since_lowest_loss == epoch_margin:
            print(f'\nEarly stopping triggered. Best loss was {lowest_loss:.4f}')
            break
 
    return loss_history

if __name__ == '__main__':

    data_path = Path('/cs/student/project_msc/2025/rai/mdecastr/Object Detection/project_sn_de_Castro_Ribeiro_jardim/dataset/RGB_depth_annotations')
    all_samples = []
    classes = []

    # Expects the dataset directory to be in the working directory, with the individual student datasets inside the general dataset folder
    for student in sorted(data_path.iterdir()):
        if not student.is_dir(): continue
        for folder in student.iterdir():
            if not folder.is_dir(): continue
            for gesture_dir in folder.iterdir():
                if not gesture_dir.is_dir(): continue
                    
                gesture_label = gesture_dir.name
                if gesture_label not in classes:
                    classes.append(gesture_label)
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
    random.seed(42) 
    random.shuffle(all_samples)

    total_size = len(all_samples)
    train_end = int(0.8 * total_size)
    val_end = int(0.85 * total_size)

    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:]

    print(f"Split sizes -> Train: {len(train_samples)}, Val: {len(val_samples)}")

    augmentations_options = [4] 
    activation_options = [nn.LeakyReLU, nn.ReLU]
    learning_rate_options = [1e-3, 1e-4]

    best_f1 = 0.0
    best_config = {}

    all_experiments = []

    val_dataset = HandGestureDataset(val_samples, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    print("Starting Hyperparameter Grid Search...")

    # Hyperparameter search, over augmentations, activation functions and learning rates
    for augs in augmentations_options:
        train_transform = MultiTaskAugmentation(is_train=True)
        train_dataset = HandGestureDataset(train_samples, transform=train_transform, num_augmentations=augs)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
        
        for act_func in activation_options:
            for lr in learning_rate_options:

                print(f"\n Testing Config: Augs={augs}, Act={act_func.__name__}, LR={lr}")
                
                # Instantiate the model 
                model = MultiTaskRGBDNet(num_classes=10, activation=act_func).to('cuda:0')
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Train the model
                loss_history = train_model(model, train_loader, optimizer, num_epochs=500, epoch_margin=6, device='cuda:0')
                
                # Evaluate on Validation Set
                results = evaluate_model(model, val_loader, device='cuda:0')
                
                current_run_data = {
                    'Augmentations': augs,
                    'Activation': act_func.__name__,
                    'Learning Rate': lr,
                    'Results': results,
                    'Loss_History': loss_history
                }
                
                # Append it to master list
                all_experiments.append(current_run_data)
                
                torch.save(model.state_dict(), f'model_{act_func.__name__}, {lr}.pth')

                # Track the best model
                if results['f1_macro'] > best_f1:
                    best_f1 = results['f1_macro']
                    best_config = current_run_data
                    # Save the physical weights of the best model
                    torch.save(model.state_dict(), 'best_model_gridsearch.pth')
                
                # Clean up GPU memory 
                del model
                del optimizer
                torch.cuda.empty_cache()
                gc.collect()


    print("\n\n Grid Search Complete! ")
    print(f"Best F1 Score: {best_f1:.4f}")

    with open('all_gridsearch_metrics.json', 'w') as f:
        json.dump(all_experiments, f, indent=4)
    print("Saved all experiment metrics and loss histories to 'all_gridsearch_metrics_resnet.json'")