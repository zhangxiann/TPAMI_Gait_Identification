from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model_cnn import Net
from gait_cnn_dataset import EV_Gait_IMG_DATASET
import torch
import argparse
import os
import sys
import torch.nn.functional as F
sys.path.append("..")
from config import Config


learning_rate = 0.00001
# learning_rate = 0.0000003

parser = argparse.ArgumentParser()
# parser.add_argument("--name", required=True, help="The path of dataset")
parser.add_argument(
    "--img_type", default='four_channel', help="The num of event image channels"
)

parser.add_argument("--epoch", default=50, type=int, help="The number of epochs")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--cuda", default="0", help="The GPU ID")

args = parser.parse_args()
img_type_dict={
    'time_only_two_channel':{'channel_num':2, 'file': Config.two_channels_time_file},
    'counts_only_two_channel':{'channel_num':2, 'file': Config.two_channels_counts_file},
    'counts_and_time_two_channel':{'channel_num':2, 'file': Config.two_channels_counts_and_time_file},
    'four_channel':{'channel_num':4, 'file': Config.four_channels_file},
}
device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename=Config.cnn_train_log_path.format(args.img_type), level=logging.DEBUG)


model = Net(img_type_dict[args.img_type]['channel_num'])
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_dataset = EV_Gait_IMG_DATASET(os.path.join(Config.image_dir, img_type_dict[args.img_type]['file']), train=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

test_dataset = EV_Gait_IMG_DATASET(os.path.join(Config.image_dir, img_type_dict[args.img_type]['file']), train=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)



for epoch in range(1, args.epoch):
    model.train()
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(train_dataloader, desc="Epoch: {}".format(epoch))):
        # with autograd.detect_anomaly():
        input, label = data
        # w*h*c -> c*h*w
        input = input.transpose(1, 3).float()
        input = input.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        end_point = model(input)
        loss = F.cross_entropy(end_point, label)
        pred = end_point.max(1)[1]
        total += len(label)
        correct += pred.eq(label).sum().item()
        loss.backward()
        optimizer.step()
    logging.info("Epoch: {} Acc: {}".format(epoch, float(correct) / total))

# test
model.eval()
correct = 0
total = 0
for index, data in enumerate(test_dataloader):
    input, label = data
    input = input.transpose(1, 3).float()
    input = input.to(device)
    label = label.to(device)
    end_point = model(input)
    pred = end_point.max(1)[1]
    total += len(label)
    correct += pred.eq(label).sum().item()


logging.info("test acc is {}".format(float(correct) / total))
print("test acc is {}".format(float(correct) / total))


torch.save(model.state_dict(), os.path.join(Config.cnn_model_name.format(args.img_type, epoch)))
