import sys
sys.path.append("..")

import torch 
import os
import json
from loss import mIoU, pixel_accuracy, Weighted_Pixel_Wise_CrossEntropyLoss
from Model.baseline_model import UNet
from Model.Res_UNet import Res_UNet
import numpy as np
from sklearn.metrics import r2_score, precision_recall_fscore_support
from collections import defaultdict
from post_process import get_single_class_counting_number


num_fold = 10
model_name  = 'UNet_v2'
dataset_name = 'Kaggle'
model_folder_path = '/home/y_yang/scratch/model_weight/{}/{}'.format(dataset_name, model_name)
train_id_path = '/home/y_yang/scratch/training_ids/{}/{}'.format(dataset_name, model_name)
state_save_folder = '/home/y_yang/scratch/train_log/{}/{}'.format(dataset_name, model_name)

if model_name == 'UNet_v2':
    model = UNet(model_type='v2', num_in_channels=3, num_out_channels=2)
elif model_name == 'Res_UNet':
    model = Res_UNet(num_in_channels=3, num_out_channels=2, apply_final_layer=True)


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

    model_weight_path = '/home/y_yang/scratch/model_weight/{}/{}/{}_weighted_ce_loss_{}_{}.pt'.format(dataset_name, model_name, model_name, dataset_name, i)
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

            pred = model(image)

            iou = mIoU(pred, mask, n_classes=1)
            acc = pixel_accuracy(pred, mask)

            preds = pred.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)

            pred_count_5 = get_single_class_counting_number(preds[0], 5)
            pred_count_15 = get_single_class_counting_number(preds[0], 15)
            pred_count_30 = get_single_class_counting_number(preds[0], 30)

            precision, recall, f1, _ = precision_recall_fscore_support(preds.reshape(-1), mask.detach().cpu().numpy().reshape(-1))

            seg_state['mIoU'][i].append(iou)
            seg_state['Acc'][i].append(acc)
            seg_state['F1'][i].append(f1[1])
            seg_state['Recall'][i].append(recall[1])
            seg_state['Precision'][i].append(precision[1])

            count_state[valid_data_folder] = [count, pred_count_5, pred_count_15, pred_count_30]  
        else:
            continue


json.dump(seg_state, open(os.path.join(state_save_folder, "valid_seg_res_{}_{}.txt".format(dataset_name, model_name)),'w'))  
json.dump(count_state, open(os.path.join(state_save_folder, "valid_count_res_{}_{}.txt".format(dataset_name, model_name)),'w'))