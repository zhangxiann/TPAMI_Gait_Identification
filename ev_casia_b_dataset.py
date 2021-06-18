import collections
import os.path as osp
import os
import errno
import numpy as np
import glob
import scipy.io as sio
import torch
import torch.utils.data
from torch_geometric.data import Data, DataLoader, Dataset
import os.path as osp

angle_list = [
    "000",
    "018",
    "036",
    "054",
    "072",
    "090",
    "108",
    "126",
    "144",
    "162",
    "180",
]


def files_exist(files):
    return all([osp.exists(f) for f in files])


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, angle_id=None):
        if not angle_id is None:
            self.angle = angle_list[angle_id]
        else:
            self.angle = None
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, "*.mat"))
        # print(filenames)
        file = [f.split("/")[-1] for f in filenames]
        if self.angle:
            file = list(filter(lambda x: True if self.angle in x else False, file))
        # print(file)
        return file

    @property
    def processed_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, "*.mat"))
        file = [f.split("/")[-1] for f in filenames]
        if self.angle:
            file = list(filter(lambda x: True if self.angle in x else False, file))
        saved_file = [f.replace(".mat", ".pt") for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        if files_exist(self.raw_paths):
            return
        print("No found data!!!!!!!")

    def process(self):
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            # print(raw_path)
            content = sio.loadmat(raw_path)
            feature = torch.tensor(content["feature"])[:, 0:1].float()
            edge_index = torch.tensor(
                np.array(content["edges"]).astype(np.int32), dtype=torch.long
            )
            pos = torch.tensor(np.array(content["pseudo"]), dtype=torch.float32)
            label_idx = torch.tensor(int(content["label"]), dtype=torch.long)
            data = Data(
                x=feature, edge_index=edge_index, pos=pos, y=label_idx.unsqueeze(0)
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            saved_name = raw_path.split("/")[-1].replace(".mat", ".pt")
            torch.save(data, osp.join(self.processed_dir, saved_name))

    def get(self, idx):
        data = torch.load(osp.join(self.processed_paths[idx]))
        return data


if __name__ == "__main__":
    pass
