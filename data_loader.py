
import torch
import torchio as tio
import numpy as np
import pandas as pd
import nibabel as nib

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from transforms import get_transforms
from scipy.ndimage import zoom

class BTRCDataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 ratio,
                 transform=None,
                 train=True
                ):
        
        #2x
        if train:
            self.df = pd.concat([df, df]).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.ratio=ratio
        self.image_size=image_size

    def __len__(self):
            return self.df.shape[0]

    def __getitem__(self, index):
            row = self.df.iloc[index]
            image=nib.load(f'../nifti/{row.hospital_id}_imd.nii.gz').get_fdata()
            
            #change size
            image = zoom(image, (self.ratio, self.ratio, 1))
            
            image = image.astype(np.float32)
            image /= 255
            
            image=(image-0.45)/0.225
            # 3 channels
            images=np.stack((image, image, image), axis=0)
            
            """
            if self.transform is not None:
                images = self.transform(image=images)['image']
            """
            #print (images.shape)

            
            # change dim D x C x H x W 
            images = images.transpose(3, 0, 1,2)
            
            n=36
            
            if len(images)>n or len(images)==n:
                images=images[len(images)//2-n//2:len(images)//2+n//2]
            else:
                images=np.concatenate((images, np.zeros((n-len(images) ,3, self.image_size, self.image_size))), axis=0)
                
            #change C x D x H x W
            images = images.transpose(1, 0, 2,3)

            label = np.zeros(1).astype(np.float32)
            label[0] = row['hemorrhage']
            
            subject = tio.Subject(
                t1=tio.ScalarImage(tensor=images),
                target=torch.tensor(label)
            )
                           
            if self.transform is not None:
                subject_trans = self.transform(subject)
            else:
                subject_trans = subject
                
            return subject_trans
        
def get_train_loaders(fold, config):
    df_train=pd.read_csv('./HT_ver1.csv')
    train_idx = np.where((df_train['fold'] != fold))[0]
    valid_idx = np.where((df_train['fold'] == fold))[0]

    df_this  = df_train.loc[train_idx]
    df_valid = df_train.loc[valid_idx]
    
    transforms=get_transforms()

    dataset_train = BTRCDataset(df_this , config['image_size'], ratio=0.5, transform=transforms[0], train=True)
    dataset_valid = BTRCDataset(df_valid, config['image_size'], ratio=0.5, transform=transforms[1], train=False)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], sampler=RandomSampler(dataset_train), num_workers=6, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=config['batch_size'], sampler=SequentialSampler(dataset_valid), num_workers=6, pin_memory=True)
    
    return train_loader, valid_loader
