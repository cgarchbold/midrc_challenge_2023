config={
    'wandb': True,
    'experiment_name': 'ResNet-18, No-Augment (50 epochs)', # Must be an acceptable directory name!
    'model':'ResNet-18',
    'imgnet_pretrained':True,
    'epochs':50,
    'augment' : False, # True: Applies augmentations
    'cropping_augmentation':False, # True: Applies cropping augmentations, False: w/o cropping augmentations
    'batch_size': 16,
    'learning_rate': 0.001,
    'root_dir': '../data/resized_224X224',
    'annotations_path': 'MIDRC mRALE Mastermind Training Annotations_2079_20230428.csv'
}