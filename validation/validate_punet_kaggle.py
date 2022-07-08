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
dataset_name = 'Kaggle'
model_folder_path = '/home/y_yang/scratch/model_weight/Kaggle/Probabilistic_UNet'
train_id_path = '/home/y_yang/scratch/training_ids/Kaggle/Probabilistic_UNet'
model = ProbabilisticUnet(num_in_channels=3, num_classes=2, latent_dim=4, criterion=Weighted_Pixel_Wise_CrossEntropyLoss())
kaggle_count = json.load(open('/home/y_yang/scratch/2018_Data_Science_Bowl/kaggle_counts.txt')) 

device = torch.device('cuda')
seg_state= {'mIoU':[[] for i in range(num_fold)], 'Acc':[[] for i in range(num_fold)], 
            'F1':[[] for i in range(num_fold)], 'Recall':[[] for i in range(num_fold)], 
            'Precision':[[] for i in range(num_fold)]}

count_state = defaultdict()

for i, train_ids in enumerate(sorted(os.listdir(train_id_path))):
    if model_name in train_ids and dataset_name in train_ids:
        valid_ids_path = os.path.join(train_id_path, train_ids+"/train_valid_ids.txt")
        valid_ids = json.load(open(valid_ids_path))['valid']
    else:
        continue

    model_weight_path = '/home/y_yang/scratch/model_weight/{}/{}/{}_weighted_ce_loss_Kaggle_{}.pt'.format(dataset_name, model_name, model_name, i)
    checkpoint = torch.load(model_weight_path)
    model.load_state_dict(checkpoint)
    model.to(device)

    for j, valid_data_folder in enumerate(sorted(os.listdir('/home/y_yang/scratch/2018_Data_Science_Bowl/stage1_train'))):
        if j in valid_ids:
            count = kaggle_count[valid_data_folder]
            image_path = os.path.join('/home/y_yang/scratch/2018_Data_Science_Bowl/stage1_train', valid_data_folder+'/images/{}.npy'.format(valid_data_folder))
            mask_path = os.path.join('/home/y_yang/scratch/2018_Data_Science_Bowl/stage1_train', valid_data_folder+'/masks/{}.npy'.format(valid_data_folder))
            
            mask = np.load(mask_path).astype(np.uint8)
            mask = torch.tensor(mask).unsqueeze(0)
            mask = mask.to(device)

            image = np.load(image_path).astype(np.float32)/255.
            image = torch.tensor(image)
            image = image.to(device)
            image = image.permute(2,0,1).unsqueeze(0)
            model.eval()
            
            res = torch.zeros((1, 2, 256, 256), device=device)

            for _ in range(20):
                model(image, training=False)
                res += model.sample(testing=True)

            pred = res / 20

            iou = mIoU(pred, mask, n_classes=1)
            acc = pixel_accuracy(pred, mask)

            preds = pred.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            
            pred_count = get_single_class_counting_number(preds[0])
            precision, recall, f1, _ = precision_recall_fscore_support(preds.reshape(-1), mask.detach().cpu().numpy().reshape(-1))

            seg_state['mIoU'][i].append(iou)
            seg_state['Acc'][i].append(acc)
            seg_state['F1'][i].append(f1[1])
            seg_state['Recall'][i].append(recall[1])
            seg_state['Precision'][i].append(precision[1])

            count_state[valid_data_folder] = [count, pred_count]  
        else:
            continue


json.dump(seg_state, open(os.path.join('/home/y_yang/scratch/train_log/Kaggle/Probabilistic_UNet', "valid_seg_res_kaggle_unet.txt"),'w'))  
json.dump(count_state, open(os.path.join('/home/y_yang/scratch/train_log/Kaggle/Probabilistic_UNet', "valid_count_res_kaggle_unet.txt"),'w'))  