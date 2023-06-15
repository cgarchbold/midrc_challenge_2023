config={
    'wandb': True,
    'experiment_name': 'Contrastive Pretrained ResNet-50, Augment(Random Rotation(20), HorizF(0.1)) without cropping (Epoch=100), cos scheduler + adamw', # Must be an acceptable directory name!
    'model':'ResNet-50',
    'imgnet_pretrained':False,
    'contrastive_pretraining':True,
    'model_freezing': -1, # Freezing techniques (#-1:None , #1: Last layer trainable,  )
    'epochs':100,
    'augment' : True, # True: Applies augmentations
    'cropping_augmentation':False, # True: Applies cropping augmentations, False: w/o cropping augmentations
    'batch_size': 16,
    'learning_rate': 0.001,
    'scheduler':'cos',
    'scheduler_warmup':1,
    'optim': 'adamw',
    'root_dir': '../data/resized_224X224',
    'annotations_path': 'MIDRC mRALE Mastermind Training Annotations_2079_20230428.csv'
}