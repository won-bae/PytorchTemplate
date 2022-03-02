import numpy as np
from PIL import Image
from torchvision import datasets


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, mode, download, logger, transform=None):
        train = True if mode == 'train' else False
        super(CIFAR10, self).__init__(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        input_dict = {
            'inputs': img,
            'labels': target,
            'indices': index
        }
        return input_dict


