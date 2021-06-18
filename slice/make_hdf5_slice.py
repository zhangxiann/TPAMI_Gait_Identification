# coding:utf-8
# 把 txt 数据转换为 image，保存到 hdf5 中


import h5py
import numpy as np
from tqdm import tqdm
import concurrent.futures
import os
import sys
import glob

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import enviroments
from convert_event_to_channel_image import generate_two_channels_count
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--name", required=True, help="The path of dataset")
parser.add_argument(
    "--window", default=100, type=int, help="The window interval time length"
)
args = parser.parse_args()



def rescale(data):
    # data[:,0]=data[:,0]*0.9
    data[:, 1] = np.round(data[:, 1] * 0.95)
    data[:, 2] = np.round(data[:, 2] * 0.95)
    # data[:,3]=data[:,3]*0.9
    return data



def generate_image(file_path, data_name):
    data = np.loadtxt(file_path)
    image = generate_two_channels_count(data)
    return image, data_name

if __name__ == '__main__':

    # 保存的文件位置
    if not os.path.exists(enviroments.image_dir_2020):
        os.makedirs(enviroments.image_dir_2020)
    target_file = os.path.join(enviroments.image_dir_2020,  "two_channels_interval_"+str(args.window)+".hdf5")

    f = h5py.File(target_file, "w")
    # 原始数据所在文件夹
    origin_dir = os.path.join(enviroments.data_dir_2020, 'txt_' + str(args.window))

    dir = ["train", "test"]
    # scenes = ['indoor_night1', 'indoor_night2']
    path_list = []
    data_names = []
    for scene in dir:
        persons = os.listdir(os.path.join(origin_dir, scene))
        for person in persons:
            txt_files = glob.glob(os.path.join(origin_dir, scene, person, '*.txt'))
            # txt_files = os.listdir(os.path.join(origin_dir, scene, person))
            for txt_file in txt_files:
                # 读取的是原始数据
                file_path = os.path.join(origin_dir, scene, person, txt_file)
                path_list.append(file_path)
                # data = np.loadtxt(filepath)
                # data=rescale(data)
                # data_name: scene_xh_-2020_09_12_10_43_29.txt

                data_name = scene + "_" + person + "_" + os.path.basename(txt_file)
                data_names.append(data_name)
                # image = generate_inputs(data)
                # f.create_dataset(name=data_name, data=image)


    with concurrent.futures.ProcessPoolExecutor() as executor:
        for image, data_name in executor.map(generate_image, path_list, data_names):
            f.create_dataset(name=data_name, data=image)
            print(data_name)

    f.close()

