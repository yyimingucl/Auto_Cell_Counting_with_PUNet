import sys
sys.path.append("..")

import torch 
import os
import json
from loss import mIoU, pixel_accuracy, Weighted_Pixel_Wise_CrossEntropyLoss
from Model.Probabilistic_UNet import ProbabilisticUnet
import numpy as np
from sklearn.metrics import r2_score, precision_recall_fscore_support
from collections import defaultdict
from post_process import get_single_class_counting_number


num_fold = 10
model_name  = 'Probabilistic_UNet'

dataset_name = 'CoNIC'
model_folder_path = '/home/y_yang/scratch/model_weight/{}/{}'.format(dataset_name, model_name)
train_id_path = '/home/y_yang/scratch/training_ids/{}/{}'.format(dataset_name, model_name)
state_save_folder = '/home/y_yang/scratch/train_log/{}/{}'.format(dataset_name, model_name)


model = ProbabilisticUnet(num_in_channels=3, num_classes=7, latent_dim=14, criterion=Weighted_Pixel_Wise_CrossEntropyLoss())

conic_count = np.genfromtxt('/home/y_yang/scratch/CoNIC_Data/conic_counts.csv', delimiter=',', skip_header = 1).astype(np.float32)


device = torch.device('cuda')
seg_state= {'mIoU':[[] for i in range(num_fold)], 'Acc':[[] for i in range(num_fold)], 
            'F1':[[] for i in range(num_fold)], 'Recall':[[] for i in range(num_fold)], 
            'Precision':[[] for i in range(num_fold)]}

count_state = defaultdict()


all_images = np.load('/home/y_yang/scratch/CoNIC_Data/conic_images.npy').astype(np.float32)
all_masks = np.load('/home/y_yang/scratch/CoNIC_Data/conic_masks.npy')[:,:,:,1].astype(np.float32)

for i, train_ids in enumerate(sorted(os.listdir(train_id_path))):
    print('[INFO] Testing Fold {}'.format(i))
    if model_name in train_ids and dataset_name in train_ids:
        valid_ids_path = os.path.join(train_id_path, train_ids+"/train_valid_ids.txt")
        valid_ids = json.load(open(valid_ids_path))['valid']
    else:
        continue
    
    model_weight_path = '/home/y_yang/scratch/model_weight/{}/{}/{}_weighted_ce_loss_{}_{}.pt'.format(dataset_name, model_name, model_name, dataset_name, i)
    checkpoint = torch.load(model_weight_path)
    model.load_state_dict(checkpoint)
    model.to(device)

    for j in valid_ids:
        
        count = conic_count[j]

        image = all_images[j] / 255.
        image = torch.tensor(image)
        image = image.to(device)
        image = image.permute(2,0,1).unsqueeze(0)
        
        
        mask = all_masks[j]
        mask = torch.tensor(mask).unsqueeze(0)
        mask = mask.to(device)

        model.eval()

        res = torch.zeros((1, 7, 256, 256), device=device)

        for _ in range(20):
            model(image, training=False)
            res += model.sample(testing=True)

        pred = res / 20
        
        

        iou = mIoU(pred, mask, n_classes=6)
        acc = pixel_accuracy(pred, mask)

        preds = pred.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)

        mask = mask.detach().cpu().numpy()
        
        temp_precision = []
        temp_recall = []
        temp_f1 = []
        temp_count = [count]
    

        for k in range(6):
            each_class_pred = np.where(preds==k+1, 1, 0)
            each_class_mask = np.where(mask==k+1, 1, 0)

            label_s = np.unique(each_class_mask)
            if len(label_s) == 1:
                f1 = [1, np.nan]
                recall = [1, np.nan]
                precision = [1, np.nan]
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(each_class_mask.reshape(-1), each_class_pred.reshape(-1), labels=label_s)
            
            temp_precision.append(precision[1])
            temp_recall.append(recall[1])
            temp_f1.append(f1[1])

            # pred_count_10 = get_single_class_counting_number(each_class_pred[0, 16:-16]*255, 10)
            # temp_count.append(pred_count_10)
            

        seg_state['mIoU'][i].append(iou)
        seg_state['Acc'][i].append(acc)
        seg_state['F1'][i].append(temp_f1)
        seg_state['Recall'][i].append(temp_recall)
        seg_state['Precision'][i].append(temp_precision)

        # count_state[j] = temp_count



json.dump(seg_state, open(os.path.join(state_save_folder, "valid_seg_res_{}_{}.txt".format(dataset_name, model_name)),'w'))  
json.dump(count_state, open(os.path.join(state_save_folder, "valid_count_res_{}_{}.txt".format(dataset_name, model_name)),'w'))