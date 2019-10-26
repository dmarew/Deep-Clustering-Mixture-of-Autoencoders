from common import *

class AutoEncoderDataset(Dataset):

    def __init__(self, data, aspect_image=None):
        self.data = data
        self.aspect_image = aspect_image

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.aspect_image is None:
            return self.data[idx], self.data[idx]
        return self.data[idx], self.aspect_image

class ATGDataset(Dataset):
    """ ATG dataset """

    def __init__(self, dataset, image_size=64):

        super(ATGDataset, self).__init__()

        self.dataset   = dataset
        self.transformer = transforms.Compose([
                           transforms.Resize(image_size + 2),
                           transforms.CenterCrop(image_size),
                           transforms.ToTensor()])

    def __getitem__(self, index):
        """
        returns
        """
        image = Image.open(self.dataset[index])
        image = self.transformer(image)
        return image
    def __len__(self):
        """length of dataset"""
        return len(self.dataset)
