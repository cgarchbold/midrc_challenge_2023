import torch
import os
import cross_fold
from get_model import create_model
from sklearn.metrics import cohen_kappa_score
from midrc_dataset import midrc_challenge_dataset
from config import config
from plotting import plot_train_metrics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import wandb
import statistics as stat

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
            f'Fold {fold_number} avg_train_loss': avg_loss.item(),
            f'Fold {fold_number} avg_val_loss': avg_vloss.item(),
            f'Fold {fold_number} avg_train_kappa': avg_train_kappa,
            f'Fold {fold_number} avg_val_kappa': avg_val_kappa,
            f'Fold {fold_number} epoch': e
        }
        if config['wandb']==True:
            wandb.log(metrics[e])

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join('experiments',config['experiment_name'],'saved_models', 'modelsave_fold_{}'.format(fold_number))
            torch.save(model.state_dict(), model_path)

    return metrics


'''
    Trains all folds in the dataset.
'''
def train_folds():

    if config['wandb']==True:   
        wandb.init(
        entity='chill-cga',
        group=config['model'],
        name=config['experiment_name'],
        # set the wandb project where this run will be logged
        project="midrc-challenge-2023",
        # track hyperparameters and run metadata
        config=config
    )

    folds = cross_fold.create_folded_datasets("../data/resized_224X224/label_info/labels.json")

    root_dir = config['root_dir']

    annotations_file = config['annotations_path']

    ex_directory = os.path.join('experiments',config['experiment_name'])
    if not os.path.exists(ex_directory):
        os.makedirs(ex_directory)

    models_directory = os.path.join(ex_directory,'saved_models')
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    plots_directory = os.path.join(ex_directory,'plots')
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config['augment']:
        if config['cropping_augmentation']==True:
            transform = transforms.Compose([
                transforms.RandomRotation(20),                           # Randomly rotate the image within -20 to +20 degrees
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),# Randomly crop and resize the image to 224x224 pixels
                transforms.RandomHorizontalFlip(0.1),                       # Randomly flip the image horizontally
                transforms.ToTensor()
            ])
        elif config['cropping_augmentation']==False:
            transform = transforms.Compose([
                transforms.RandomRotation(20),                           # Randomly rotate the image within -20 to +20 degrees
                transforms.RandomHorizontalFlip(0.1),                       # Randomly flip the image horizontally
                transforms.ToTensor()
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([transforms.ToTensor()])

    saved_metrics = []

    # train over all folds
    for f_i,fold in enumerate(folds):
        print("FOLD: ",f_i+1)
        train_list, val_list = fold

        model = create_model(config=config)
        model.to(device)

        # TODO: Choose optimizer in config
        optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])

        # TODO: Choose loss in config
        criterion = torch.nn.MSELoss()

        train_dataset = midrc_challenge_dataset(root_dir, annotations_file, transform, fp_list = train_list)
        val_dataset = midrc_challenge_dataset(root_dir, annotations_file, val_transform, fp_list = val_list)
        
        train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle=True)

        #Training per fold
        metrics = train(config['epochs'],model,device,train_loader,val_loader, criterion, optimizer, f_i+1) # Folds will be started from 1 instead of 0
        saved_metrics.append(metrics)
    

    # Save results to table in wandb
    to_save_val_results=[]
    all_fold_loss=[]
    all_fold_kappa=[]
    for cur_fold in range(len(saved_metrics)):
        cur_fold_best_loss=1e10
        cur_fold_best_kappa=0
        for cur_fold_epoch in range(len(saved_metrics[cur_fold])):
            if saved_metrics[cur_fold][cur_fold_epoch]['Fold '+str(cur_fold+1)+' avg_val_loss']<cur_fold_best_loss:
                cur_fold_best_loss=saved_metrics[cur_fold][cur_fold_epoch]['Fold '+str(cur_fold+1)+' avg_val_loss']
                cur_fold_best_kappa=saved_metrics[cur_fold][cur_fold_epoch]['Fold '+str(cur_fold+1)+' avg_val_kappa']
        to_save_val_results.append([str(cur_fold+1),cur_fold_best_loss,cur_fold_best_kappa])
        all_fold_loss.append(cur_fold_best_loss)
        all_fold_kappa.append(cur_fold_best_kappa)
    to_save_val_results.append(['mean',stat.mean(all_fold_loss),stat.mean(all_fold_kappa)])
    to_save_val_results.append(['stdev',stat.stdev(all_fold_loss),stat.stdev(all_fold_kappa)])
    columns=['Fold','Loss','Kappa']   
    if config['wandb']==True:
        val_table=wandb.Table(columns=columns,data=to_save_val_results)
        wandb.log({"Val Table": val_table})

    #Save + plot train statistics
    plot_train_metrics(folds, saved_metrics, ex_directory)
    fp = os.path.join(ex_directory,"train_metrics.pkl")
    with open(fp, "wb") as file:
        pickle.dump(saved_metrics, file)


if __name__ == "__main__":
    train_folds()
    if config['wandb']==True:
        wandb.finish()