config={
    'experiment_name': 'No_Augment', # Must be an acceptable directory name!
    'model':'ResNet-50',
    'imgnet_pretrained':True,
    'epochs':1,
    'augment' : False,
    'batch_size': 16,
    'learning_rate': 0.001,
    'root_dir': '../data/resized_224X224',
    'annotations_path': 'MIDRC mRALE Mastermind Training Annotations_2079_20230428.csv'
}