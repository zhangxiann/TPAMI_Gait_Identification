# coding:utf-8
# convet event streams to image like representation
from itertools import repeat

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
from config import Config
import convert_event_to_channel_image_tool


def generate_wrap_function(generate_function, file_path, data_name):
    data = np.loadtxt(file_path)
    image = generate_function(data)
    return image, data_name


def generate_event_img(hdf5_path, generate_function, data_names, path_list):
    f = h5py.File(hdf5_path, "w")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for image, data_name in executor.map(generate_wrap_function, repeat(generate_function), path_list, data_names):
            f.create_dataset(name=data_name, data=image)
            # print(data_name)

    f.close()


if __name__ == '__main__':

    # 保存的文件位置
    if not os.path.exists(Config.image_dir):
        os.makedirs(Config.image_dir)
    # 原始数据所在文件夹
    origin_dir = os.path.join(Config.data_dir, 'origin')

    path_list = []
    data_names = []
    for train_test in os.listdir(origin_dir):
        persons = os.listdir(os.path.join(origin_dir, train_test))
        for person in persons:
            txt_files = glob.glob(os.path.join(origin_dir, train_test, person, '*.txt'))
            # txt_files = os.listdir(os.path.join(origin_dir, scene, person))
            for txt_file in txt_files:
                # 读取的是原始数据
                file_path = os.path.join(origin_dir, train_test, person, txt_file)
                path_list.append(file_path)
                data_name = train_test + "_" + person + "_" + os.path.basename(txt_file)
                data_names.append(data_name)
                #break

    generate_event_img(Config.two_channels_counts_file, convert_event_to_channel_image_tool.generate_two_channels_count, data_names, path_list)
    generate_event_img(Config.two_channels_time_file, convert_event_to_channel_image_tool.generate_two_channels_time, data_names, path_list)
    generate_event_img(Config.four_channels_file, convert_event_to_channel_image_tool.generate_four_channels, data_names, path_list)
    generate_event_img(Config.two_channels_counts_and_time_file, convert_event_to_channel_image_tool.generate_two_channels_counts_and_time, data_names, path_list)

    # f = h5py.File(Config.two_channels_counts_file, "w")
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for image, data_name in executor.map(generate_two_channels_count, path_list, data_names):
    #         f.create_dataset(name=data_name, data=image)
    #         # print(data_name)
    #
    # f.close()