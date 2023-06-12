config={
    'wandb': True,
    'experiment_name': 'ResNet-50, Augment with cropping (Epoch=100)', # Must be an acceptable directory name!
    'model':'ResNet-50',
    'imgnet_pretrained':True,
    'epochs':100,
    'augment' : True, # True: Applies augmentations
    'cropping_augmentation':True, # True: Applies cropping augmentations, False: w/o cropping augmentations
    'batch_size': 16,
    'learning_rate': 0.001,
    'root_dir': '../data/resized_224X224',
    'annotations_path': 'MIDRC mRALE Mastermind Training Annotations_2079_20230428.csv'
}