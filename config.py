config={
    'wandb': True,
    'experiment_name': 'ResNet-50, Augment w/o cropping', # Must be an acceptable directory name!
    'model':'ResNet-50',
    'imgnet_pretrained':True,
    'epochs':50,
    'augment' : True,
    'cropping_augmentation':False,
    'batch_size': 16,
    'learning_rate': 0.001,
    'root_dir': '../data/resized_224X224',
    'annotations_path': 'MIDRC mRALE Mastermind Training Annotations_2079_20230428.csv'
}