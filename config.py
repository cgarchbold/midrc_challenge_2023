config={
    'wandb': True,
    'experiment_name': 'XRV Pretrained, DenseNet-121, Normalized sigmoid', # Must be an acceptable directory name!
    'model':'DenseNet-121',
    'imgnet_pretrained':False,
    'contrastive_pretraining':False,
    'model_freezing': -1, # Freezing techniques (#-1:None , #1: fc trainable,  )
    'epochs':100,
    'augment' : True, # True: Applies augmentations
    'cropping_augmentation':False, # True: Applies cropping augmentations, False: w/o cropping augmentations
    'batch_size': 8,
    'torchxrayvision_pretrained':True,
    'learning_rate': 0.001,
    'scheduler':'cos',
    'scheduler_warmup':1,
    'optim': 'adamw',
    'normalized_output':True, # True: model's prediction will be in the range of [0,1], False: model's prediction will be in the range of [0,24]
    'root_dir': '../data/resized_224X224',
    'annotations_path': 'MIDRC mRALE Mastermind Training Annotations_2079_20230428.csv'
}