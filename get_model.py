import torchvision
import torch

def create_model(config):
    if config['model']=='ResNet-50':
        if config['imgnet_pretrained']:
            model =  torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        else:
            model =  torchvision.models.resnet50()
        # ResNet 50 model (1 channel input, Sigmoid Output)
        model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3), bias=False)
        model.fc = torch.nn.Sequential(
                    torch.nn.Linear(2048, 512, bias=True),
                    torch.nn.Linear(512, 256, bias=True),
                    torch.nn.Linear(256, 1, bias = True),
                    torch.nn.Sigmoid())
        return model
    
    if config['model']=='ResNet-18':
        if config['imgnet_pretrained']:
            model =  torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
        else:
            model =  torchvision.models.resnet18()
        # ResNet 50 model (1 channel input, Sigmoid Output)
        model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3), bias=False)
        model.fc = torch.nn.Sequential(
                        torch.nn.Linear(512, 256, bias=True),
                        torch.nn.Linear(256, 1, bias=True),
                        torch.nn.Sigmoid())
        return model