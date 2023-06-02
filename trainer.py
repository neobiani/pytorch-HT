import os

import torch
import torch.nn as nn
import torchvision.models as models

from data_loader import get_train_loaders
from losses import get_loss_criterion
from metrics import get_evaluation_metric
from model import get_model
from utils import create_optimizer, create_lr_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torchio as tio
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, f1_score


def train_epoch(model, loader, optimizer, device, loss_criterion, scaler):

    model.train()
    train_loss = []
    
    bar=tqdm(loader)
    
    for data in bar:
        
        inputs = data['t1'][tio.DATA].to(device)
        target = data['target'].to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(inputs.half())
            loss = loss_criterion(logits, target)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss

def val_epoch(model, loader, device, loss_criterion, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    
    with autocast():

        with torch.no_grad():
            for data in tqdm(loader):
                inputs = data['t1'][tio.DATA].to(device)
                target = data['target'].to(device)
                
                logits = model(inputs.half())

                loss = loss_criterion(logits, target)

                pred = logits.sigmoid().detach()
                LOGITS.append(logits)
                PREDS.append(pred)
                TARGETS.append(target)

                val_loss.append(loss.detach().cpu().numpy())
            val_loss = np.mean(val_loss)
            
        #print (LOGITS)

        LOGITS = torch.cat(LOGITS).cpu().numpy()
        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()

        PREDS=np.where(np.isnan(PREDS), 0, PREDS)

        print (np.round(PREDS).sum())
        print (TARGETS.sum())
        acc = (np.round(PREDS) == TARGETS).mean() * 100.
        auc = roc_auc_score (TARGETS, PREDS)
        f1 = f1_score(TARGETS, np.round(PREDS))

        print('acc', acc, 'auc', auc, 'f1', f1)

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, auc, f1

def train_(model, loaders, optimizer, scheduler, config, loss_criterion, fold):
    
    val_min = 0.8
    kernel_type=config['kernel_type']
    n_epochs=config['n_epochs']
    scaler=GradScaler()

    
    best_file = f'{kernel_type}_best_fold{fold}.pt'
    for i in range(1):
        with open(f'log_{kernel_type}.txt', 'a') as appender:
                    appender.write(f'fold:{fold}\n')

        for epoch in range(1, n_epochs+1):
            print(time.ctime(), 'Epoch:', epoch)
            
            train_loss = train_epoch(model, loaders[0], optimizer, config['device'], loss_criterion, scaler)
            
            scheduler.step(epoch-1)

            if epoch <1 :
                pass
            else:
                val_loss, acc, auc, f1 = val_epoch(model, loaders[1], config['device'], loss_criterion)
                #test_acc, test_auc = test_epoch(test_loader)

                #content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f},  auc: {(auc):.5f}, test_acc: {(test_acc):.5f},  test_auc: {(test_auc):.5f}'
                content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f},  auc: {(auc):.5f}, f1: {(f1):.5f}'
                print(content)
                with open(f'log_{kernel_type}.txt', 'a') as appender:
                    appender.write(content + '\n')

                if (f1-val_loss) > val_min:
                    print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(val_min, val_loss))
                    checkpoint = {
                                            'model': model.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'scaler': scaler.state_dict()}
                    torch.save(checkpoint, best_file)
                    val_min = (f1-val_loss)

    checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict()}
    torch.save(checkpoint, os.path.join(f'{kernel_type}_final_fold{fold}.pt'))

def create_trainer(fold, config):
    # Create the model
    model = get_model(config['model_type'], config['out_dim'])
    # use DataParallel if more than 1 GPU available
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        model.cuda()

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(fold, config)

    # Create the optimizer
    optimizer = create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config, optimizer)

    #trainer_config = config['trainer']
    
    # Create trainer
    resume = config['Resume']

    return train_(model=model, optimizer=optimizer, scheduler=lr_scheduler, loss_criterion=loss_criterion, loaders=loaders, fold=fold, config=config)