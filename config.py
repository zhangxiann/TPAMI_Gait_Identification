import os

class Config():
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]
    downsample_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day/downsample')
    graph_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day/graph')
    graph_train_dir = os.path.join(graph_dir, 'train')
    graph_test_dir = os.path.join(graph_dir, 'test')
    log_dir = os.path.join(rootPath, 'log')
    train_log_path = os.path.join(log_dir, 'train.log')
    model_dir = os.path.join(rootPath, 'trained_model')
    model_path = os.path.join(model_dir, 'EV_Gait_3DGraph_epoch_{}.pkl')