from common import *
class AutoEncoderDataset(Dataset):

    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]
