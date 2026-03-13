import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import calc_bbox_iou, calc_mask_metrics, gather_test_samples
from model import MultiTaskRGBDNet
from dataloader import HandGestureDataset
from evaluate import evaluate_model

def evaluate_model(model, val_loader, device='cuda:1'):
    '''
    Performs inference on the model and calculates key metrics
    '''
    model.eval()
    
    all_preds_cls = []
    all_targets_cls = []
    
    total_bbox_iou = 0.0
    total_det_acc_05 = 0.0
    
    total_mask_iou = 0.0
    total_mask_dice = 0.0
    
    total_samples = 0
    
    with torch.no_grad(): 
        for batch in val_loader:
            rgb, depth, target_cls, target_box, target_mask = [t.to(device) for t in batch]
            
            pred_cls, pred_box, pred_mask = model(rgb, depth)
            
            _, predicted_classes = torch.max(pred_cls, 1)
            all_preds_cls.extend(predicted_classes.cpu().numpy())
            all_targets_cls.extend(target_cls.cpu().numpy())
            
            bbox_ious = calc_bbox_iou(pred_box, target_box)
            total_bbox_iou += bbox_ious.sum().item()
            total_det_acc_05 += (bbox_ious > 0.5).sum().item() 

            mask_ious, mask_dices = calc_mask_metrics(pred_mask, target_mask)
            total_mask_iou += mask_ious.sum().item()
            total_mask_dice += mask_dices.sum().item()
            
            total_samples += target_cls.size(0)

    results = {
        'cls_accuracy': 100.0 * np.sum(np.array(all_preds_cls) == np.array(all_targets_cls)) / total_samples,
        'f1_macro': f1_score(all_targets_cls, all_preds_cls, average='macro'),
        'bbox_iou_mean': total_bbox_iou / total_samples,
        'det_acc_05': 100.0 * total_det_acc_05 / total_samples,
        'mask_iou_mean': total_mask_iou / total_samples,
        'mask_dice_mean': total_mask_dice / total_samples
    }
    
    print("\n--- Validation Results ---")
    print(f"Classification Acc: {results['cls_accuracy']:.2f}% | F1 Macro: {results['f1_macro']:.4f}")
    print(f"BBox Mean IoU:      {results['bbox_iou_mean']:.4f}  | Det Acc @ 0.5: {results['det_acc_05']:.2f}%")
    print(f"Mask Mean IoU:      {results['mask_iou_mean']:.4f}  | Dice Coeff:    {results['mask_dice_mean']:.4f}")
    
    return results

def main():
    print('getting the paths right')
    TEST_DATA_PATH = '/cs/student/project_msc/2025/rai/mdecastr/Object Detection/project_sn_de_Castro_Ribeiro_jardim/dataset/test_set'
    WEIGHTS_PATH = 'model_LeakyReLU, 0.001.pth'
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16

    print('GETTING THE TEST SAMPLES')
    test_samples, num_classes = gather_test_samples(TEST_DATA_PATH)
    
    if len(test_samples) == 0:
        print("Error: No samples found. Please check TEST_DATA_PATH.")
        return

    test_dataset = HandGestureDataset(test_samples, transform=None)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )
    
    print("\nInitializing model architecture...")
    model = MultiTaskRGBDNet(num_classes=num_classes, activation=nn.LeakyReLU)
    
    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    print("\nStarting evaluation on test set...")
    test_results = evaluate_model(model, test_loader, device=DEVICE)
    
    print("\n--- Final Test Set Results ---")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()