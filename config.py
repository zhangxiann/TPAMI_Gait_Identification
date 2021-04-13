import os

class Config():
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]
    downsample_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day/downsample')
    data_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day')

    graph_dir = os.path.join(data_dir, 'graph')
    graph_train_dir = os.path.join(graph_dir, 'train')
    graph_test_dir = os.path.join(graph_dir, 'test')
    log_dir = os.path.join(rootPath, 'log')
    graph_train_log_path = os.path.join(log_dir, 'graph_train.log')
    cnn_train_log_path = os.path.join(log_dir, 'cnn_train_{}.log')
    model_dir = os.path.join(rootPath, 'trained_model')
    gcn_model_path = os.path.join(model_dir, 'EV_Gait_3DGraph_epoch_{}.pkl')

    image_dir = os.path.join(data_dir, 'image')
    two_channels_counts_file = os.path.join(image_dir, 'two_channels_counts.hdf5')
    four_channels_file = os.path.join(image_dir, 'four_channels.hdf5')
    two_channels_time_file = os.path.join(image_dir, 'two_channels_time.hdf5')
    two_channels_counts_and_time_file = os.path.join(image_dir, 'two_channels_counts_and_time.hdf5')
    cnn_model_path = os.path.join(model_dir, 'EV-Gait-IMG_epoch_{}.pkl')
