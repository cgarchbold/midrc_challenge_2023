import torchvision
import torch
def create_model(config):
    if config['model']=='ResNet-50':
        model =  torchvision.models.resnet50(pretrained=config['imgnet_pretrained'])
        # ResNet 50 model (1 channel input, )
        model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3), bias=False)
        model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1, bias=True), torch.nn.Sigmoid())
        return model