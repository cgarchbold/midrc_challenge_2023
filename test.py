import torch
import os
import cross_fold
from get_model import create_model
from sklearn.metrics import cohen_kappa_score
from midrc_dataset import midrc_challenge_dataset
import config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def test(model,device,dataloader):
    model.eval()
    with torch.no_grad():
        squared_error = 0
        absolute_error = 0
        s_percent_error = 0
        kappa = 0
        kappa_list_y1 = []
        kappa_list_y2 = []

        for data in dataloader:
            inputs, labels = data

            outputs = model(inputs.to(device))
            labels=labels/24.0

            # For error metrics we normalize labels and outputs to 0-1
            squared_error += (outputs - labels.to(device))**2
            absolute_error += torch.abs(outputs-labels.to(device))
            s_percent_error += torch.abs((outputs-labels.to(device))/((labels.to(device) + outputs)/2))


            outputs=outputs*24.0
            labels=labels*24.0
            outputs=torch.round(outputs)
            outputs=outputs.data.cpu().numpy()
            labels=labels.data.cpu().numpy()
            outputs=outputs.flatten()
            labels=labels.flatten()

            kappa_list_y1.extend(labels)
            kappa_list_y2.extend(outputs)


        kappa = cohen_kappa_score(kappa_list_y1,kappa_list_y2,weights='quadratic')
            
        #returns mae and rmse
        return absolute_error/len(dataloader), torch.sqrt(squared_error/len(dataloader)), s_percent_error/len(dataloader), kappa
    
def test_folds():

    folds = cross_fold.create_folded_datasets("../data/label_info/labels.json") #TODO: define in config

    root_dir = config['root_dir']

    annotations_file = config['annotations_path']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([transforms.ToTensor()])

    ex_directory = os.path.join('experiments',config['experiment_name'])
    
    #Testing Loop
    for f_i,fold in enumerate(folds):

        train_list, val_list = fold

        val_dataset = midrc_challenge_dataset(root_dir, annotations_file, val_transform, fp_list = val_list)
        val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=True)

        model = create_model(config)
        model_pth = os.path.join('models','modelsave_fold_{}'.format(f_i))
        model.load_state_dict(torch.load(model_pth))
        model.to(device)
        model.eval()

        avg_mae, avg_rmse, avg_smape, kappa = test(model,device,val_loader)

        #TODO: Save to file instead, save as dict?
        print("Fold: ",f_i+1, " MAE:", avg_mae.item() , " RMSE: ", avg_rmse.item(), " sMAPE: ", avg_smape.item(), " Kappa: ", kappa)


if __name__ == "__main__":
    test_folds()