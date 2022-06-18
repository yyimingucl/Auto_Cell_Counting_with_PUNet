import os
import torch 
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Model import Res_UNet, UNet, FCRN_A_BASE
from torch.utils.tensorboard import SummaryWriter
from loss import pixel_accuracy, mIoU
from parameter import hyper_param

root_path = os.path.dirname(os.path.abspath(__file__))
date = datetime.now()
date = date.strftime('%c')

config = {'model':'Res_UNet', 
          'num_epochs':hyper_param.num_epochs, 
          'loss':'weighted_ce_loss', 
          'lr':hyper_param.lr,
          'model_save_folder':'model_weight',
          'log_save_folder':'train_log',
          'dataset': 'CoNIC',  # Fluorescent
          'use_loss_weight':True,
          'train/valid_split_rate':0.1, 
          'batch_size': hyper_param.batch_size, 
          'optimizer': 'Adam'}

run_name = '{}_{}_{}'.format(date, config['model'], config['dataset'])
log_save_path = os.path.join(config['log_save_folder'], run_name) 

model_name = '{}_{}_{}'.format(config['model'], config['loss'], config['dataset'])
model_save_path = os.path.join(config['model_save_folder'], model_name)


#Set up tensorboard
writer = SummaryWriter(log_save_path)


#Set up Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set up Dataset
if config['dataset'] == 'CoNIC':
    from dataloader import CoNIC_DATASET

    num_class = 6
    print('[INFO] Building DataLoader for CoNIC Dataset')

    image_file = '/data/data/conic_images.npy'
    images = np.load(image_file).astype(np.float32)
    mask_file = '/data/data/conic_masks.npy'
    masks = np.load(mask_file)[:,:,:,1].astype(np.int32)

    dataset = CoNIC_DATASET(images, masks)

    if config['use_loss_weight']:
        loss_weight_file = '/data/data/conic_loss_weights.npy'
        loss_weight = np.load(loss_weight_file).astype(np.float32)
        dataset.obtain_loss_weights(loss_weight)
    else:
        loss_weight = None

elif config['dataset'] == 'Kaggle':
    from dataloader import Kaggle_DATASET

    num_class = 1
    print('[INFO] Building DataLoader for 2018 Data Science Bowl Dataset')
    
    base_path = '/data/data-science-bowl-2018/stage1_train'
    sample_sets = os.listdir(base_path)

    dataset = Kaggle_DATASET(base_path, sample_sets, config['use_loss_weight'])

elif config['dataset'] == 'Fluorescent':
    from dataloader import FluoRescent_DATASET

    num_class = 1
    print('[INFO] Building DataLoader for Fluorescent Microscopy Dataset')

    base_path = '/data/Fluorescent_Data'
    sample_sets = os.listdir('/data/Fluorescent_Data/all_images/images')

    dataset = FluoRescent_DATASET(base_path, sample_sets, config['use_loss_weight'])

else:
    raise NameError('[WARNING] Not Known DataSet Name') 


# Split Train and Valid Dataset and Create DataLoader
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(config['train/valid_split_rate'] * dataset_size))
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=valid_sampler)
print("[INFO] Number of training/validing patches:", (len(train_indices),len(valid_indices)))

# Set up Model
if config['model'] == 'Res_UNet':
    print('[INFO] Create Model: Res_UNet')
    from Model.Resnet_based_model import Res_UNet
    model = Res_UNet(num_in_channels=3, num_out_channels=num_class+1, apply_final_layer=True)
model.to(device)

# Set up Loss
if config['loss'] == 'weighted_ce_loss':
    print('[INFO] Loss Function: Weighted Cross Entropy Loss')
    from loss import Weighted_Pixel_Wise_CrossEntropyLoss
    criterion = Weighted_Pixel_Wise_CrossEntropyLoss()


# Set up optimizer
if config['optimizer'] == 'Adam':
    print('[INFO] Optimizer: Adam, Learning Rate: {}'.format(config['lr']))
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=config['lr'])

elif config['optimizer'] == 'SGD':
    print('[INFO] Optimizer: SGD, Learning Rate: {}'.format(config['lr']))
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=config['lr'])    
# scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)



print('[INFO] Start Training')
for epoch in range(config['num_epochs']): 
    print(' ')
    print('[INFO] {}th Epoch Start'.format(epoch+1))

    # -----------------------------  Training ------------------------------------ #
    model.train()
    train_iou_score = 0
    train_accuracy = 0
    train_loss = 0
    
    for package in train_loader:

        data = package[0].to(device)
        target = package[1].to(device)

        if len(package) == 2:
            loss_weight = None
        elif len(package) == 3:
            loss_weight = package[2].to(device)
        
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        loss = criterion(preds=output, targets=target, pixelwise_weights=loss_weight)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        iou = mIoU(output, target, n_classes=num_class)
        acc = pixel_accuracy(output, target)
        
        train_loss += loss.item()
        train_iou_score += iou
        train_accuracy += acc

   
    # -----------------------------  Validting  ------------------------------------ #
    model.eval()
    valid_iou_score = 0
    valid_accuracy = 0
    valid_loss = 0

    for package in valid_loader:

        data = package[0].to(device)
        target = package[1].to(device)

        if len(package) == 2:
            loss_weight = None
        elif len(package) == 3:
            loss_weight = package[2].to(device)
        
        with torch.no_grad():
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            loss = criterion(output, target, loss_weight)
            iou = mIoU(output, target, n_classes=num_class)
            acc = pixel_accuracy(output, target)

        valid_loss += loss
        valid_iou_score += iou
        valid_accuracy += acc

    # -------------------------------- Logging Stats ------------------------------ #
    train_loss = train_loss / train_loader.__len__()
    train_iou_score = train_iou_score / train_loader.__len__()
    train_accuracy = train_accuracy / train_loader.__len__()

    valid_loss = valid_loss / valid_loader.__len__()
    valid_iou_score = valid_iou_score / valid_loader.__len__()
    valid_accuracy = valid_accuracy / valid_loader.__len__()

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('mIoU/Train', train_iou_score, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

    writer.add_scalar('Loss/Test', valid_loss, epoch)
    writer.add_scalar('mIoU/Test', valid_iou_score, epoch)
    writer.add_scalar('Accuracy/Test', valid_accuracy, epoch) 

    
    # scheduler.step(valid_loss)
    # save model if validation loss has decreased  
    print('[INFO] Validation loss: {:.5f}.'.format(valid_loss))
    print('[INFO] Validation mIOU: {:.5f}'.format(valid_iou_score))
    print('[INFO] Validation Accuracy: {:.5f}'.format(valid_accuracy))
    torch.save(model.state_dict(), model_save_path)    
    print(' ')
