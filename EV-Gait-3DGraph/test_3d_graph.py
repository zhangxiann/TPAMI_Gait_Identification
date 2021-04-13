# -- coding: utf-8 --**
# test the Run EV-Gait-3DGraph model



import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import argparse

from tqdm import tqdm

from model_3d_graph import Net
import os
import logging
import sys
sys.path.append("..")
from config import Config


from ev_gait_3d_graph_dataset import EV_Gait_3DGraph_Dataset







if __name__ == '__main__':
    if not os.path.exists(Config.log_dir):
        os.makedirs(Config.log_dir)

    if not os.path.exists(Config.model_dir):
        os.makedirs(Config.model_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--model_path", default="Test_EV_Gait_3DGraph.pkl", help="The GPU ID")
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(filename=Config.graph_train_log_path, level=logging.DEBUG)

    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(Config.model_dir, args.model_path)))
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.96, 0.999])])
    # test_data_aug = T.Compose([T.Cartesian(cat=False)])
    test_dataset = EV_Gait_3DGraph_Dataset(
        Config.graph_test_dir, transform=test_data_aug
    )
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=2, pin_memory=True)


    # test
    model.eval()
    correct = 0
    total = 0

    for index, data in enumerate(tqdm(test_loader)):
        data = data.to(device)
        end_point = model(data)
        pred = end_point.max(1)[1]
        total += len(data.y)
        correct += pred.eq(data.y).sum().item()

    logging.info("test acc is {}".format(float(correct) / total))
    print("test acc is {}".format(float(correct) / total))


