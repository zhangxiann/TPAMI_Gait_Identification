from torch.utils.data import Dataset
import h5py



class EV_Gait_IMG_DATASET(Dataset):
    def __init__(self, data_path, train=True):
        self.data = h5py.File(data_path, "r")
        if train:
            self.keys = [key for key in self.data.keys() if key.startswith('train')]
        else:
            self.keys = [key for key in self.data.keys() if key.startswith('test')]

    def __len__(self):
        return self.keys.__len__()

    def __getitem__(self, index):
        # self.keys[index] == 'Train_9_00058'
        return self.data.get(self.keys[index])[:], int(self.keys[index].split('_')[1])