# coding:utf-8
# 将  event stream 分割为多个片段
import os
import numpy as np
import sys
import concurrent.futures
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import enviroments
import glob
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--name", required=True, help="The path of dataset")
parser.add_argument(
    "--window", default=50, type=int, help="The window interval time length"
)
args = parser.parse_args()

# 过滤 stream 前后的空白片段，
# window_length: 滑动窗口时间长度
def filter_slience(filepath, window_length=50, threshold=500):
    file = np.loadtxt(filepath)
    window_time = window_length * 1e3
    theta = threshold
    t_start = file[0, 0]
    t_end = file[-1, 0]

    t = t_start
    while t + window_time < t_end:
        count = 0
        # 计算第一个时间窗口内的 event 数量
        for i in file[:, 0]:
            if i < t + window_time and i > t:
                count += 1
        # 如果这个时间窗口内的 event 数量大于 theta，则退出循环
        if count > theta:
            filter_t_start = t
            break

        # 否则继续下一个时间窗口
        t += window_time / 2
    # print(filter_t_start)
    t = t_end

    while t - window_time > t_start:
        count = 0
        # 反转 event stream, 从最后一个 window 开始计算,
        for i in reversed(file[:, 0]):
            if i > t - window_time and i < t:
                count += 1
        # 如果这个时间窗口内的 event 数量大于 theta，则退出循环
        if count > theta:
            filter_t_end = t
            break
        # 否则继续下一个时间窗口
        t -= window_time / 2
    # print(filter_t_end)
    filter_result = file[
        np.where((file[:, 0] > filter_t_start) & (file[:, 0] < filter_t_end)), :
    ]
    return np.squeeze(filter_result, 0)

def split_slice(file_path, window_length=args.window):

    data = filter_slience(file_path)
    window_time = window_length * 1e3
    window_start = data[0, 0]
    end = data[-1, 0]
    # print("start = "+str(window_start))
    # print("end = "+str(end))
    t = window_start
    window_end = window_start + window_time
    slice_num = 0
    file_paths = file_path.split(os.path.sep)
    # 新目录: /home/iot/Disk1/ZhangXian/DATA/DVS128_2020/txt_100
    target_dir = os.path.sep.join(file_paths[:-3]) + '_' + str(window_length)
    while t + window_time <= end:
        slice_start_id = (np.abs(data[:, 0] - window_start)).argmin()
        slice_end_id = (np.abs(data[:, 0] - window_end)).argmin()
        # print(str(slice_end_id- slice_start_id))
        slice_data = data[slice_start_id:slice_end_id, :]

        new_file_path = target_dir + os.path.sep + os.path.sep.join(file_paths[-3:])
        (path, ext) = os.path.splitext(new_file_path)
        new_file_path = path+'_'+str(slice_num)+ext
        parent_dir = os.path.dirname(new_file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        np.savetxt(new_file_path, slice_data, fmt='%d')
        slice_num = slice_num+1
        # 否则继续下一个时间窗口
        t += window_time
        window_start = t
        window_end = window_start + window_time
    return file_path, slice_num



# 保存的文件位置

# 原始数据所在文件夹
origin_dir = os.path.join(enviroments.data_dir_2020,'txt')


dir = ["train", "test"]
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


all_file_num = 0

window_length=args.window
target_dir = origin_dir + '_' + str(window_length)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
with concurrent.futures.ProcessPoolExecutor() as executor:
    for file_path, slice_num in executor.map(split_slice, path_list):
        all_file_num = all_file_num + slice_num
        print(file_path)

with open('slice.log', "a+") as file:
    file.write("slice: "+str(args.window)+", all_file_couts: " + str(all_file_num) + ", average: " + str(all_file_num/len(path_list))+'\n')



# split_slice('D:\\Project\\event_algorithm\\Graph-data\\DVS128_2020\\txt\\train\\1\\1.txt')

# split_slice('/home/iot/Disk1/ZhangXian/DATA/DVS128_2020/txt/train/1/1.txt')



























