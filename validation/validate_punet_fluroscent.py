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
dataset_name = 'Flourscent'
model_folder_path = '/home/y_yang/scratch/model_weight/{}/{}'.format(dataset_name, model_name)
train_id_path = '/home/y_yang/scratch/training_ids/{}/{}'.format(dataset_name, model_name)
state_save_folder = '/home/y_yang/scratch/train_log/{}/{}'.format(dataset_name, model_name)
model = ProbabilisticUnet(num_in_channels=3, num_classes=2, latent_dim=4, criterion=Weighted_Pixel_Wise_CrossEntropyLoss())



flourscent_count = json.load(open('/home/y_yang/scratch/Flourscent_Data/flourscent_counts.txt')) 


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

    model_weight_path = '/home/y_yang/scratch/model_weight/{}/{}/{}_weighted_ce_loss_Flourscent_{}.pt'.format(dataset_name, model_name, model_name, i)
    checkpoint = torch.load(model_weight_path)
    model.load_state_dict(checkpoint)
    model.to(device)


    for j, valid_data in enumerate(sorted(os.listdir('/home/y_yang/scratch/Flourscent_Data/all_images/images'))):
        if j in valid_ids:
            count = flourscent_count[valid_data]
            image_path = os.path.join('/home/y_yang/scratch/Flourscent_Data/all_images/images/{}'.format(valid_data))
            mask_path = os.path.join('/home/y_yang/scratch/Flourscent_Data/all_masks/masks/{}'.format(valid_data))
            
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

            mask = mask.detach().cpu().numpy()
            # pred_count_5 = get_single_class_counting_number(preds[0], 5)
            pred_count_15 = get_single_class_counting_number(preds[0], 15)
            # pred_count_30 = get_single_class_counting_number(preds[0], 30)
            label_s = np.unique(mask)
            if len(label_s) == 1:
                f1 = [1, np.nan]
                recall = [1, np.nan]
                precision = [1, np.nan]
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(mask.reshape(-1), preds.reshape(-1), labels=label_s)

            seg_state['mIoU'][i].append(iou)
            seg_state['Acc'][i].append(acc)
            seg_state['F1'][i].append(f1[1])
            seg_state['Recall'][i].append(recall[1])
            seg_state['Precision'][i].append(precision[1])

            count_state[valid_data] = [count, pred_count_15]  

        else:

            continue


json.dump(seg_state, open(os.path.join(state_save_folder, "valid_seg_res_{}_{}.txt".format(dataset_name, model_name)),'w'))  
json.dump(count_state, open(os.path.join(state_save_folder, "valid_count_res_{}_{}.txt".format(dataset_name, model_name)),'w'))