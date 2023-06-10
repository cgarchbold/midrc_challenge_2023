import torch
import os
import cross_fold
from get_model import create_model
from sklearn.metrics import cohen_kappa_score
from midrc_dataset import midrc_challenge_dataset
import config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

'''
    Train on 1 fold (1 dataset) with the input settings

    Returns: dict of training metrics
             best validation model is saved to \models
'''
def train(epochs,model,device, train_loader, val_loader, criterion, optimizer, fold_number):
    metrics={}
    best_vloss = 1_000_000.
    for e in range(epochs):
        print('EPOCH {}:'.format(e + 1))
        running_loss = 0.
        last_loss = 0.
        running_train_kappa=0.0
        avg_train_kappa=0.0
        model.train(True)

        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs.to(device))
            labels=labels/24.0

            # Compute the loss and its gradients
            loss = criterion(outputs, labels.float().to(device).unsqueeze(1))
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            outputs=outputs*24.0
            labels=labels*24.0
            
            outputs=torch.round(outputs)
            outputs=outputs.data.cpu().numpy()
            
            labels=labels.data.cpu().numpy()
            outputs=outputs.flatten()
            labels=labels.flatten()
            kappa_score_train=cohen_kappa_score(labels,outputs,weights='quadratic')
            running_train_kappa+=kappa_score_train
            # Gather data
            running_loss += loss
        
        avg_loss = running_loss/(i+1)
        avg_train_kappa=running_train_kappa/(i+1)

        
        running_vloss = 0.0
        running_kappa = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs.to(device))
                vlabels=vlabels/24.0
                vloss = criterion(voutputs, vlabels.float().to(device).unsqueeze(1))
                voutputs=voutputs*24.0
                vlabels=vlabels*24.0
                voutputs=torch.round(voutputs)
                voutputs=voutputs.data.cpu().numpy()
                vlabels=vlabels.data.cpu().numpy()
                voutputs=voutputs.flatten()
                vlabels=vlabels.flatten()
                kappa_score=cohen_kappa_score(vlabels,voutputs,weights='quadratic')
                running_kappa+=kappa_score
                running_vloss += vloss
                

        avg_vloss = running_vloss / (i + 1)
        avg_val_kappa= running_kappa / (i + 1)

        
        print('LOSS train {} valid {}, Kappa train {} valid {}'.format(avg_loss,avg_vloss,avg_train_kappa,avg_val_kappa))
        metrics[e] = {      
            'avg_loss': avg_loss.item(),
            'avg_vloss': avg_vloss.item(),
            'avg_train_kappa': avg_train_kappa,
            'avg_val_kappa': avg_val_kappa
        }   

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join('models','modelsave_fold_{}'.format(fold_number))
            torch.save(model.state_dict(), model_path)

    return metrics


'''
    Trains all folds in the dataset.
'''
def train_folds():

    folds = cross_fold.create_folded_datasets("../data/resized_224X224/label_info/labels.json")

    root_dir = config['root_dir']

    annotations_file = config['annotations_path']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config['augment']:
        transform = transforms.Compose([
            transforms.RandomRotation(20),                           # Randomly rotate the image within -20 to +20 degrees
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),# Randomly crop and resize the image to 224x224 pixels
            transforms.RandomHorizontalFlip(0.1),                       # Randomly flip the image horizontally
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([transforms.ToTensor()])

    saved_metrics = []

    for f_i,fold in enumerate(folds):
        print("FOLD: ",f_i+1)
        train_list, val_list = fold

        model = create_model(config=config)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters())

        criterion = torch.nn.MSELoss()

        train_dataset = midrc_challenge_dataset(root_dir, annotations_file, transform, fp_list = train_list)
        val_dataset = midrc_challenge_dataset(root_dir, annotations_file, val_transform, fp_list = val_list)
        
        train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle=True)

        #Training per fold
        metrics = train(config['epochs'],model,device,train_loader,val_loader, criterion, optimizer, f_i)
        saved_metrics.append(metrics)