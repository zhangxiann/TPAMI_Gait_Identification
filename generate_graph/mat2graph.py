# -- coding: utf-8 --**
# convert downsample file to graph

import numpy as np
import os
import scipy.io as sio
from config import Config
import concurrent.futures


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--type", default=None, help="scene type")

args = parser.parse_args()



def calculate_edges(data, r=5):
    # threshold of radius
    d = 32
    # scaling factor to tune the difference between temporal and spatial resolution
    alpha = 1
    beta = 1
    data_size = data.shape[0]
    # max number of edges is 1000000,
    edges = np.zeros([1000000, 2])
    # get t, x,y
    points = data[:, 0:3]
    row_num = 0
    for i in range(data_size - 1):
        count = 0
        distance_matrix = points[i + 1 : data_size + 1, 0:3]
        distance_matrix[:, 1:3] = distance_matrix[:, 1:3] - points[i, 1:3]
        distance_matrix[:, 0] = distance_matrix[:, 0] - points[i, 0]
        distance_matrix = np.square(distance_matrix)
        distance_matrix[:, 1:3] *= alpha
        distance_matrix[:, 0] *= beta
        # calculate the distance of each pair of events
        distance = np.sqrt(np.sum(distance_matrix, axis=1))
        index = np.where(distance <= r)
        # save the edges
        if index:
            index = index[0].tolist()
            for id in index:
                edges[row_num, 0] = i
                edges[row_num + 1, 1] = i
                edges[row_num, 1] = int(id) + i + 1
                edges[row_num + 1, 0] = int(id) + i + 1
                row_num = row_num + 2
                count = count + 1
        if count > d:
            break
    edges = edges[~np.all(edges == 0, axis=1)]
    edges = np.transpose(edges)
    return edges


# get polarity as the feature of the node
def extract_feature(data):
    data_size = data.shape[0]
    feature = np.zeros([data_size, 1])
    for i in range(data_size):
        if data[i, 3] == 1:
            feature[i, 0] = +1
        else:
            feature[i, 0] = -1
    return feature


def extract_position(data):
    data_size = data.shape[0]
    position = np.zeros([data_size, 3])
    for i in range(data_size):
        position[i, :] = data[i, 0:3]
    return position


def generate_graph(origin_path, target_path, label):
    file = sio.loadmat(origin_path)
    data = file["points"]
    feature = extract_feature(data)
    position = extract_position(data)
    edges = calculate_edges(data, 5)
    # if the number of edges is 0 or less than 10, skip this sample
    if edges.shape[1]<10:
        # view this file
        print(origin_path+" : "+str(edges.shape[1]))
        return
    save_data = {"feature": feature, "pseudo": position, "edges": edges, "label": label}


    sio.savemat(target_path, save_data)
    return target_path

def main():
    origin_path_list=[]
    label_list=[]
    target_path_list=[]
    # iterate train and test
    for train_test in os.listdir(Config.downsample_dir):
        # iterate each person
        for person in os.listdir(os.path.join(Config.downsample_dir, train_test)):
            # make corresponding person graph dir
            if not os.path.exists(os.path.join(Config.graph_dir, train_test, "raw")):
                os.makedirs(os.path.join(Config.graph_dir, train_test, "raw"))
            index =0
            for file in os.listdir(os.path.join(Config.downsample_dir, train_test, person)):
                index = index+1
                if index>2:
                    break
                origin_path_list.append(os.path.join(Config.downsample_dir, train_test, person, file))
                label_list.append(person)
                # faltten the graph files in one directory
                target_path_list.append(os.path.join(Config.graph_dir,  train_test,"raw", person+"_"+file))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(generate_graph, origin_path_list, target_path_list, label_list):
            #print(result)
            pass

if __name__ == "__main__":
    main()
    print("genetate graph complete")
