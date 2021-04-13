# coding:utf-8
import h5py
import numpy as np
import os
from tqdm import tqdm
import concurrent.futures

# convert event stream to image like representation

# four channels
# two channels for count, two channels for time
def generate_four_channels(events_data):
    image = np.zeros([128, 128, 4])

    # timestamp of first event
    min_time = events_data[0, 0]
    # timestamp of last event
    max_time = events_data[-1, 0]
    # events_data: (t,x,y,p)
    for i in range(events_data.shape[0]):
        # wether polarity is positive or negtive
        if events_data[i, 3] == 1:
            # calculate the counts for positive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 1] = (
                    image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 1] + 1
            )
            # calculate the temporal characteristics for positive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 3] = (events_data[i, 0] - min_time) / (max_time - min_time)
        else:
            # calculate the counts for negtive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 0] = (
                    image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 0] + 1
            )
            # calculate the characteristics for negtive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 2] = (
                                                                                   events_data[i, 0] - min_time
                                                                           ) / (max_time - min_time)

    return image


# two channels (counts and time temporal characteristics)
def generate_two_channels_counts_and_time(events_data):

    image = np.zeros([128, 128, 2])
    # timestamp of first event
    min_time = events_data[0, 0]
    # timestamp of last event
    max_time = events_data[-1, 0]

    for i in range(events_data.shape[0]):
        # calculate the counts for both positive and negtive events
        image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 0] = (
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 0] + 1
        )
        # calculate the temporal characteristics for both positive and negtive events
        image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 1] = (
            events_data[i, 0] - min_time
        ) / (max_time - min_time)

    return image


# two channels of counts for positive and negtive events
def generate_two_channels_count(events_data):
    image = np.zeros([128, 128, 2])
    for i in range(events_data.shape[0]):
        if events_data[i, 3] == 1:
            # calculate the counts for positive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 1] = (
                image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 1] + 1
            )
        else:
            # calculate the counts for negtive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 0] = (
                image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 0] + 1
            )

    return image


# two channels of temporal characteristics for positive and negtive events
def generate_two_channels_time(events_data):

    image = np.zeros([128, 128, 2])

    # timestamp of first event
    min_time = events_data[0, 0]
    # timestamp of first event
    max_time = events_data[-1, 0]

    for i in range(events_data.shape[0]):

        if events_data[i, 3] == 1:
            # calculate the temporal characteristics for both positive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 0] = (
                events_data[i, 0] - min_time
            ) / (max_time - min_time)
        else:
            # calculate the temporal characteristics for both negtive events
            image[(int)(events_data[i, 1]), (int)(events_data[i, 2]), 1] = (
                events_data[i, 0] - min_time
            ) / (max_time - min_time)

    return image