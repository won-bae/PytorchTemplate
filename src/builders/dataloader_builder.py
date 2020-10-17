from torch.utils.data import DataLoader
from torchvision import transforms
from src.core.datasets import CustomDataset
from src.utils.util import normalization_params


DATASETS = {
    'custom': CustomDataset,
}

def build(mode, data_config, logger):
    data_name = data_config['name']
    root = data_config['root']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    transform_config = data_config['transform']

    train = True if mode == 'train' else False
    shuffle = data_config.get('shuffle', train)

    transform = compose_transforms(transform_config, mode)
    dataset = DATASETS[data_name](root, logger, mode, transform=transform,
                                  transform_config=transform_config)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader

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

