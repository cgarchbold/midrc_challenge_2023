from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.optim import lr_scheduler
import torch
import numpy as np
import statistics as stat
import os
import json 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torchmetrics.regression import KendallRankCorrCoef
from torch.nn import KLDivLoss
import torch


def get_scheduler(config,optimizer_ft):
    if config['scheduler']=='cos':
        scheduler_fn = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=config['epochs']//config['scheduler_warmup'], eta_min=0,verbose=True)
   
    return scheduler_fn