from torch.utils.data import DataLoader
from torchvision import transforms
from src.core.datasets import CIFAR10
from src.utils.util import normalization_params


DATASETS = {
    'cifar10': CIFAR10,
}

def build(data_config, logger):
    # Get data parameters
    data_name = data_config['name']
    root = data_config['root']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    transform_config = data_config['transform']

    if data_name not in DATASETS:
        logger.error('No data named {}'.format(data_name))

    # Load datalodaers for each mode
    dataloaders = {}
    for mode in ['train', 'val', 'test']:
        shuffle = True if mode == 'train' else False
        transform = compose_transforms(transform_config, mode)

        if data_name == 'cifar10':
            dataset = DATASETS[data_name](root=root, mode=mode, download=True,
                                          logger=logger, transform=transform)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers)
            dataloaders[mode] = dataloader

    return dataloaders

def compose_transforms(transform_config, mode):
    mean, std = normalization_params()
    image_size = transform_config['image_size']
    crop_size = transform_config['crop_size']

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return transform

