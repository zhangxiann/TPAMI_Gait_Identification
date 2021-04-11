# -- coding: utf-8 --**
# train the Run EV-Gait-3DGraph model

# nohup python -u EV-Gait-3DGraph/train_3d_graph_gait.py --type "train 2020_zx_outdoor_day1 test 2020_zx_outdoor_day2" --cuda 1 --experiment_nums 1 --result_file "gcn_result_zx_cv.log" > gcn_cv_zx.log 2>&1 &
import numpy as np


import torch
from torch import autograd
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import argparse
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
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(filename=Config.train_log_path, level=logging.DEBUG)

    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.96, 0.999])])

    train_dataset = EV_Gait_3DGraph_Dataset(
        Config.graph_train_dir, transform=train_data_aug
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    # train
    model.train()
    for epoch in range(1, 120):
        if epoch == 60:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.0001
        if epoch == 110:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.00001

        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            end_point = model(data)
            loss = F.nll_loss(end_point, data.y)
            pred = end_point.max(1)[1]
            total += len(data.y)
            correct += pred.eq(data.y).sum().item()
            loss.backward()
            optimizer.step()


        # accuracy of each epoch
        logging.info("epoch: {0}, train acc is %f".format(epoch, float(correct) / total))


    # test
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.96, 0.999])])
    test_dataset = EV_Gait_3DGraph_Dataset(
        Config.Config.graph_test_dir, transform=test_data_aug
    )
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0

        for index, data in enumerate(test_loader):
            data = data.to(device)
            end_point = model(data)
            pred = end_point.max(1)[1]
            total += len(data.y)
            correct += pred.eq(data.y).sum().item()

        logging.info("test acc is %f" % (float(correct) / total))
    state_sict = model.state_dict()
    torch.save(state_sict, Config.model_path)

