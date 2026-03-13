import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn

from resnet_training_run import HandGestureDataset, MultiTaskRGBDNet
from utils import gather_test_samples

# the order in which the classes were indexed during the training run
CLASSES = ['G07_peace', 'G10_three', 'G08_rock', 'G04_ok', 'G03_like', 
           'G09_stop', 'G01_call', 'G06_palm', 'G02_dislike', 'G05_one']


def plot_confusion_matrix(model, dataloader, device):
    """Generates and displays a confusion matrix using Seaborn."""
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\nGathering predictions for Confusion Matrix...")
    with torch.no_grad():
        for batch in dataloader:
            rgb, depth, target_cls, _, _ = [t.to(device) for t in batch]
            pred_cls, _, _ = model(rgb, depth)
            
            _, predicted = torch.max(pred_cls, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target_cls.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Gesture')
    plt.xlabel('Predicted Gesture')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("Saved confusion matrix to 'confusion_matrix.png'")

def visualize_predictions(model, dataloader, device, num_images=4):
    """Plots original RGB with Ground Truth vs Predictions side-by-side."""
    model.eval()
    
    batch = next(iter(dataloader))
    rgb, depth, target_cls, target_box, target_mask = [t.to(device) for t in batch]
    
    # performing inference
    with torch.no_grad():
        pred_cls, pred_box, pred_mask = model(rgb, depth)
        
    pred_mask_bin = (torch.sigmoid(pred_mask) > 0.5).float()
    _, predicted_classes = torch.max(pred_cls, 1)

    rgb = rgb.cpu()
    target_box = target_box.cpu()
    target_mask = target_mask.cpu()
    pred_box = pred_box.cpu()
    pred_mask_bin = pred_mask_bin.cpu()
    
    img_width, img_height = 640.0, 480.0

    fig, axes = plt.subplots(num_images, 2, figsize=(12, 5 * num_images))
    if num_images == 1:
        axes = [axes] 

    # generating images
    for i in range(min(num_images, rgb.size(0))):
        img_np = rgb[i].permute(1, 2, 0).numpy()
        
        ax_gt = axes[i][0]
        ax_gt.imshow(img_np)
        
        mask_gt_np = target_mask[i].squeeze().numpy()
        ax_gt.imshow(mask_gt_np, cmap='Reds', alpha=0.4 * mask_gt_np)
        
        gt_x_min = target_box[i][0] * img_width
        gt_y_min = target_box[i][1] * img_height
        gt_x_max = target_box[i][2] * img_width
        gt_y_max = target_box[i][3] * img_height
        
        rect_gt = patches.Rectangle((gt_x_min, gt_y_min), gt_x_max - gt_x_min, gt_y_max - gt_y_min, 
                                    linewidth=2, edgecolor='green', facecolor='none', label='GT Box')
        ax_gt.add_patch(rect_gt)
        gt_class_name = CLASSES[target_cls[i].item()]
        ax_gt.set_title(f"Ground Truth: {gt_class_name}")
        ax_gt.axis('off')

        ax_pred = axes[i][1]
        ax_pred.imshow(img_np)
        
        mask_pred_np = pred_mask_bin[i].squeeze().numpy()
        ax_pred.imshow(mask_pred_np, cmap='Blues', alpha=0.4 * mask_pred_np)

        pr_x_min = pred_box[i][0] * img_width
        pr_y_min = pred_box[i][1] * img_height
        pr_x_max = pred_box[i][2] * img_width
        pr_y_max = pred_box[i][3] * img_height
        
        rect_pred = patches.Rectangle((pr_x_min, pr_y_min), pr_x_max - pr_x_min, pr_y_max - pr_y_min, 
                                      linewidth=2, edgecolor='red', facecolor='none', label='Pred Box')
        ax_pred.add_patch(rect_pred)
        
        pred_class_name = CLASSES[predicted_classes[i].item()]
        ax_pred.set_title(f"Prediction: {pred_class_name}")
        ax_pred.axis('off')

    plt.tight_layout()
    plt.savefig('visualization_samples.png')
    plt.show()
    print("Saved visualization to 'visualization_samples.png'")

def main():
    TEST_DATA_PATH = '/cs/student/project_msc/2025/rai/mdecastr/Object Detection/project_sn_de_Castro_Ribeiro_jardim/dataset/test_set'
    WEIGHTS_PATH = 'model_LeakyReLU, 0.001.pth'
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    test_samples = gather_test_samples(TEST_DATA_PATH)
    test_dataset = HandGestureDataset(test_samples, transform=None)
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    
    print("\nLoading model...")
    model = MultiTaskRGBDNet(num_classes=len(CLASSES), activation=nn.LeakyReLU)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    plot_confusion_matrix(model, test_loader, DEVICE)
    
    print("\nGenerating sample visual predictions...")
    visualize_predictions(model, test_loader, DEVICE, num_images=4)

if __name__ == '__main__':
    main()