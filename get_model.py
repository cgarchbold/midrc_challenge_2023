import torchvision
import torch
import torchxrayvision as xrv
def create_model(config):
    if config['model']=='ResNet-50':
        if config['imgnet_pretrained']:
            model =  torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif config['torchxrayvision_pretrained']==True:
            print("Loading model from xrv")
            model=xrv.models.ResNet(weights="resnet50-res512-all")
            model=model.model
            model.fc=torch.nn.Sequential(
                    torch.nn.Linear(2048, 512, bias=True),
                    torch.nn.Linear(512, 256, bias=True),
                    torch.nn.Linear(256, 1, bias = True),
                    torch.nn.Sigmoid())
            return model
        elif config['imgnet_pretrained']==False:
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
    
    if config['model']=='ViT-L-16':
        if config['imgnet_pretrained']:
            model =  torchvision.models.vit_l_16(weights='DEFAULT')
        else:
            model =  torchvision.models.vit_l_16()
        # ViT model (1 channel input, Sigmoid Output)
        model.conv_proj = torch.nn.Conv2d(1,1024, kernel_size=(16, 16), stride=(16, 16))
        model.heads = torch.nn.Sequential(
                        torch.nn.Linear(1024, 256, bias=True),
                        torch.nn.Linear(256, 1, bias=True),
                        torch.nn.Sigmoid())
        return model
    
    if config['model']=='Swin-v2-b':
        if config['imgnet_pretrained']:
            model =  torchvision.models.swin_v2_b(weights='DEFAULT')
        else:
            model =  torchvision.models.swin_v2_b()
        # ViT model (1 channel input, Sigmoid Output)
        model.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        model.head = torch.nn.Sequential(
                        torch.nn.Linear(1024, 256, bias=True),
                        torch.nn.Linear(256, 1, bias=True),
                        torch.nn.Sigmoid())
        return model
    
def load_contrastive_pretrained_model(config, fold_number):
        if config['model']=='ResNet-50':
            model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
            model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3), bias=False)
            model.fc = torch.nn.Sequential(
                        torch.nn.Linear(2048, 512, bias=True),
                        torch.nn.Linear(512, 256, bias=True))
            model.load_state_dict(torch.load('./experiments/ResNet-50 Contrastive Pretraining Per Fold/saved_models/modelsave_fold_'+str(fold_number)+".ckpt"))
            model.fc = torch.nn.Sequential(
                    torch.nn.Linear(2048, 512, bias=True),
                    torch.nn.Linear(512, 256, bias=True),
                    torch.nn.Linear(256, 1, bias = True),
                    torch.nn.Sigmoid())
            return model
