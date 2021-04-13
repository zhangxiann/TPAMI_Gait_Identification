# coding:utf-8
# 训练 CNN 模型

import sys
import numpy as np
import h5py
import os
import random
import argparse
import glob
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import enviroments
# tf.test.is_gpu_available()

# import tensorflow as tf
#
# gpu_device_name = tf.test.gpu_device_name()
# print(gpu_device_name)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tensorflow as tf

num_classes = enviroments.subject_num
batch_size = 20
height = 128
width = 128
learning_rate = 0.0000003

parser = argparse.ArgumentParser()
# parser.add_argument("--name", required=True, help="The path of dataset")
parser.add_argument(
    "--channel_num", default=2, type=int, help="The num of event image channels"
)
# parser.add_argument("--logfile_name", default="log.txt", help="The log file name")
parser.add_argument("--epoch", default=50, type=int, help="The number of epochs")
parser.add_argument("--kernel_size", default=3, type=int, help="The kernel size")
# parser.add_argument("--type", default="two channels", help="grid image type")
parser.add_argument("--train_type", default=['indoor_day', 'outdoor_day1'], help="train scene type", nargs='+', type=str)
parser.add_argument("--test_type", default=["indoor_night"], help="test scene type", nargs='+', type=str)


parser.add_argument("--cuda", default="1", help="The GPU ID")
parser.add_argument("--result_file", default="result_single.log")
parser.add_argument("--experiment_nums", default=30, type=int, help="The number of experiment")
args = parser.parse_args()
f = h5py.File(enviroments.two_channel_file, "r")

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
print(tf.test.is_gpu_available())


def generate_inputs(dataname):
    return f[dataname]

logfile = os.path.join(enviroments.log_dir, "cnn.log")

# 原始数据所在文件夹
origin_dir = os.path.join(enviroments.data_dir, 'txt')

persons = os.listdir(os.path.join(origin_dir, args.test_type[0]))
labels=list(range(len(persons)))
person_label_dict = dict(zip(persons,labels))

all_correct = []
for ci in range(args.experiment_nums):
    tf.reset_default_graph()

    input_image = tf.placeholder(tf.float32, shape=(None, height, width, args.channel_num))

    kernel_size = args.kernel_size

    layer1_conv = tf.layers.conv2d(
        input_image,
        64,
        kernel_size=(kernel_size, kernel_size),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer1_conv",
    )
    layer2_conv = tf.layers.conv2d(
        layer1_conv,
        128,
        kernel_size=(kernel_size, kernel_size),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer2_conv",
    )
    layer3_conv = tf.layers.conv2d(
        layer2_conv,
        256,
        kernel_size=(kernel_size, kernel_size),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer3_conv",
    )
    layer4_conv = tf.layers.conv2d(
        layer3_conv,
        512,
        kernel_size=(kernel_size, kernel_size),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer4_conv",
    )

    layer5_conv1 = tf.layers.conv2d(
        layer4_conv,
        512,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer5_conv1",
    )
    layer5_conv2 = tf.layers.conv2d(
        layer5_conv1,
        512,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer5_conv2",
    )
    layer5_res = tf.add(layer5_conv2, layer4_conv, name="layer5_res")
    layer5_relu2 = tf.nn.relu(layer5_res, name="layer5_relu2")

    layer6_conv1 = tf.layers.conv2d(
        layer5_relu2,
        512,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer6_conv1",
    )
    layer6_conv2 = tf.layers.conv2d(
        layer6_conv1,
        512,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.relu,
        name="layer6_conv2",
    )
    layer6_res = tf.add(layer6_conv2, layer5_relu2, name="layer6_res")
    layer6_relu2 = tf.nn.relu(layer6_res, name="layer6_relu2")

    layer7_input = tf.reshape(layer6_relu2, [-1, 8 * 8 * 512])

    layer7_dense = tf.layers.dense(
        layer7_input, 1024, activation=tf.nn.relu, name="layer7_dense"
    )
    layer8_dense = tf.layers.dense(
        layer7_dense, 512, activation=tf.nn.relu, name="layer8_dense"
    )
    layer9_dense = tf.layers.dense(layer8_dense, num_classes, name="layer9_dense")


    input_lable = tf.placeholder(tf.float32, shape=(None, num_classes))

    probability = tf.nn.softmax(logits=layer9_dense, name="probability")

    total_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=input_lable, logits=layer9_dense
    )

    # add to validate different train size
    train_images = np.zeros([batch_size, 128, 128, args.channel_num])
    train_labels = np.zeros([batch_size, num_classes])

    # train_images = np.zeros([batch_size, 128, 128, args.channel_num])
    # train_labels = np.zeros([batch_size, num_classes])

    validate_images = np.zeros([batch_size, 128, 128, args.channel_num])
    validate_labels = np.zeros([batch_size, num_classes])

    saver = tf.train.Saver(max_to_keep=1)

    # num_list = range(1, 101)
    # # train_list=random.sample(num_list,25)
    # train_list = list(range(1, 101))
    # # train_list = range(1, 101)

    # people_list=range(21)
    # label_list=random.sample(people_list,num_classes)
    label_list = range(num_classes)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    step = 0
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    best_test_acc = 0
    total_loss_result = batch_size
    global_validation_loss = batch_size
    while step < args.epoch:
        dataset = ["train", "test"]
        batch_id = 0
        batch_index = 0

        for scene in args.train_type:
            persons = os.listdir(os.path.join(origin_dir, scene))
            for person in persons:
                txt_files = glob.glob(os.path.join(origin_dir, scene, person, '*.txt'))
                # txt_files = os.listdir(os.path.join(origin_dir, scene, person))
                for txt_file in txt_files:
                    dataname = scene + "_" + person + "_" + os.path.basename(txt_file)
                    if batch_index < batch_size - 1:
                        # 从 hdf5 中取出对应的 image
                        train_images[batch_index, :, :, :] = generate_inputs(dataname)
                        train_labels[batch_index, person_label_dict[person]] = 1
                        batch_index = batch_index + 1

                    elif batch_index == batch_size - 1:
                        train_images[batch_index, :, :, :] = generate_inputs(dataname)
                        train_labels[batch_index, person_label_dict[person]] = 1
                        # 开始训练
                        _, loss_result, prob_result = sess.run(
                            [optimizer, total_loss, probability],
                            {input_image: train_images, input_lable: train_labels},
                        )

                        accurate_num = 0.0
                        right_index = []

                        for i in range(batch_size):
                            if (
                                    np.max(prob_result[i, :])  # 取出概率中最大的值
                                    == prob_result[i, np.argmax(train_labels[i, :])]
                                    # 通过 argmax 取到最大数值的索引，然后再取出预测值中对应的数
                                    # 然后看这两个数是否同一个
                            ):
                                accurate_num = accurate_num + 1.0
                                right_index.append(np.argmax(train_labels[i, :]))
                        print("ci:", ci, ", step:", step, ", accuracy", str(accurate_num / batch_size))
                        # print(step, loss_result, accurate_num, accurate_num / batch_size)
                        # # print "Training right index"
                        # # print right_index
                        # print("===========================")

                        # print prob_result

                        train_images = np.zeros([batch_size, 128, 128, args.channel_num])
                        train_labels = np.zeros([batch_size, num_classes])

                        batch_index = 0
                        # step = step + 1

        # print(count_num)
        step = step + 1
        validate_index = 0

        # 验证
        #if step > 20:
    correct = 0.0
    for scene in args.test_type:
        persons = os.listdir(os.path.join(origin_dir, scene))
        for person in persons:
            txt_files = glob.glob(os.path.join(origin_dir, scene, person, '*.txt'))
            # txt_files = os.listdir(os.path.join(origin_dir, scene, person))
            for txt_file in txt_files:
                dataname = scene + "_" + person + "_" + os.path.basename(txt_file)

                if validate_index < batch_size - 1:
                    validate_images[validate_index, :, :, :] = generate_inputs(dataname)

                    validate_labels[validate_index, person_label_dict[person]] = 1
                    validate_index = validate_index + 1
                elif validate_index == batch_size - 1:
                    validate_images[validate_index, :, :, :] = generate_inputs(dataname)
                    validate_labels[validate_index, person_label_dict[person]] = 1

                    loss_result, prob_result = sess.run(
                        [total_loss, probability],
                        {input_image: validate_images, input_lable: validate_labels},
                    )

                    accurate_num = 0.0
                    right_index = []
                    for i in range(batch_size):
                        if (
                                np.max(prob_result[i, :])
                                == prob_result[i, np.argmax(validate_labels[i, :])]
                        ):
                            accurate_num = accurate_num + 1.0
                            right_index.append(np.argmax(validate_labels[i, :]))
                    val_acc = accurate_num / batch_size
                    correct += val_acc

                    total_loss_result = loss_result

                    validate_images = np.zeros([batch_size, 128, 128, args.channel_num])
                    validate_labels = np.zeros([batch_size, num_classes])

                    validate_index = 0
                    batch_id += 1
                    # break

    # saver.save(sess, "./model/cnn_img.ckpt")

    # 导出当前计算图的GraphDef部分
    graph_def = tf.get_default_graph().as_graph_def()
    # 查看所有节点的名称
    var_list = tf.global_variables()
    # 保存指定的节点，并将节点值保存为常数
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['softmax_cross_entropy_loss/value','input', 'label', "probability"])
    # 将计算图写入到模型文件中
    model_f = tf.gfile.GFile("cnn_model.pb", "wb")
    model_f.write(output_graph_def.SerializeToString())
    print("ci test:", ci, ", step:", step, ", accuracy", correct / batch_id)

              # train_test
        # logfile = "../log/log.txt"

    with open(logfile, "a+") as file:
        file.write("ci:" + str(ci) + ", " + ' '.join(args.train_type)+ ", " + ' '.join(args.test_type) + ": %f\n" % (best_test_acc))
    all_correct.append(best_test_acc)



result_file = os.path.join(enviroments.log_dir, args.result_file)
with open(result_file, "a+") as file:
    all_correct.sort()
    all_correct = all_correct[:30]
    # file.write("CNN :  "+' '.join(args.train_type)+ ", " + ' '.join(args.test_type) +", average" + ": %f\n" % (all_correct / args.experiment_nums))
    file.write("CNN :  "+' '.join(args.train_type)+ ", " + ' '.join(args.test_type) +", average" + ": %f, std: %f\n" % (np.average(all_correct), np.std(all_correct)))