from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root, logger, mode, transform=None):
        self.root = root
        self.logger = logger
        self.mode = mode
        self.transform =transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample.path
        label = sample.label
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)

