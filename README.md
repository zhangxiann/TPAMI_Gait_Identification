## Summary

This is the code for the paper **Event-Stream Representation for Human Gaits Identification Using Deep Neural Networks** by Yanxiang Wang, [Xian Zhang](https://blog.zhangxiann.com/), [Yiran Shen*](http://yiranshen.academic.site/), Bowen Du, Guangrong Zhao, Lizhen Cui Cui Lizhen, Hongkai Wen.

The paper can be found [here](http://academic0202101180gpyi.images.academic.site/eventstream%20representation%20early%20access%20version.pdf).



## Introduction

In this paper, We propose new event-based gait recognition approaches basing on two different representations of the event-stream, i.e., graph and image-like representations, and use Graph-based Convolutional Network (GCN) and Convolutional Neural Networks (CNN) respectively to recognize gait from the event-streams. The two approaches are termed as **EV-Gait-3DGraph** and **EV-Gait-IMG**. To evaluate the performance of the proposed approaches, we collect two event-based gait datasets, one from real-world experiments and the other by converting the publicly available RGB gait recognition benchmark CASIA-B.



If you use any of this code or data, please cite the following publication:



> ```
> @inproceedings{wang2019ev,
>   title={EV-gait: Event-based robust gait recognition using dynamic vision sensors},
>   author={Wang, Yanxiang and Du, Bowen and Shen, Yiran and Wu, Kai and Zhao, Guangrong and Sun, Jianguo and Wen, Hongkai},
>   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
>   pages={6358--6367},
>   year={2019}
> }
> ```



> ```
> @article{wang2021event,
>  title={Event-Stream Representation for Human Gaits Identification Using Deep Neural Networks},
>     author={Wang, Yanxiang and Zhang, Xian and Shen, Yiran and Du, Bowen and Zhao,     Guangrong and Lizhen, Lizhen Cui Cui and Wen, Hongkai},
>    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
>     year={2021},
>    publisher={IEEE}
>    }
>   ```



# Requirements

- Python 3.x
- Conda
- cuda
- PyTorch
- numpy
- scipy
- PyTorch Geometric
- TensorFlow
- Matlab (with **Computer Vision Toolbox** and **Image Processing Toolbox** for nonuniform grid downsample)



# Installation

- First set up an [Anaconda](https://www.anaconda.com/) environment:

  ```
  conda create -n gait python=3.7
  conda activate gait
  ```

- install the packages

  ```
  pip install tqdm
  pip install scipy
  pip install h5py
  pip install argparse
  pip install numpy
  pip install itertools
  ```

  

- Install [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) according to their documentation respectively.

  - PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
  - PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)



## Data

We use both data collected in real-world experiments(called **DVS128-Gait**) and converted from publicly available RGB gait databases(called **EV-CASIA-B**). Here we offer the code and data for the **DVS128-Gait**.


### DVS128-Gait DATASET

we use a [DVS128 Dynamic Vision Sensor](https://inivation.com/support/hardware/dvs128/) from iniVation operating at 128*128 pixel resolution.

we collect two dataset: **DVS128-Gait-Day** and **DVS128-Gait-Night**, which were collected under day and night lighting condition respectively.

For each lighting condition, we recruited  20 volunteers to contribute their data in two experiment sessions spanning over a few days. In each session, the participants were asked to repeat walking in front of the DVS128 sensor for 100 times.

- **DVS128-Gait-Day**: [https://drive.google.com/file/d/1i5znP-ozea-r8svMV9mLIJXblbJZqCBL/view?usp=sharing](https://drive.google.com/file/d/1i5znP-ozea-r8svMV9mLIJXblbJZqCBL/view?usp=sharing)
- **DVS128-Gait-Night**: https://pan.baidu.com/s/1txWR75DaAOyva6oUOJ4Kbg , extraction code: **iypf**



## Run EV-Gait-3DGraph

- download **DVS128-Gait-Day** dataset to the `data` folder, you will get **DVS128-Gait-Day.zip**, and unzip it to the `data/origin/` folder.


- event downsample using matlab:

  > 1. open Matlab
  >2. go to `matlab_downsample`
  > 3. run `main.m`. This will generate the `data/downsample` folder which contains the non-uniform octreeGrid filtering data .
  
- > or directly download the downsampled data from this link:
  >
  > https://pan.baidu.com/s/1OKKvrhid929DakSxsjT7XA , extraction code: **ceb1** 
  >
  > Then unzip it to the `data/downsample` folder.

- generate graph representation for event:

  ```
  cd generate_graph
  python mat2graph.py
  ```

  
  
- Download the pretrained model to the `trained_model` folder:

  https://pan.baidu.com/s/12VfTi5GsgKQrMoIU018NRg , extraction code: **ewzz**
  


- run EV-Gait-3DGraph model with the pretrained model:

  ```
  cd EV-Gait-3DGraph
  python test_3d_graph.py --model_path EV_Gait_3DGraph.pkl
  ```

- train EV-Gait-3DGraph from scratch:

  ```
  cd EV-Gait-3DGraph
  nohup python -u train_3d_graph.py --epoch 80 --cuda 0 > train_3d_graph.log 2>&1 
  &
  ```
  
  the traning log would be created at `log/train.log`.
  
  > parameters of **train_3d_graph.py**
  >
  > - --batch_size: default `16`
  > - --epoch: number of iterations, default `150`
  > - --cuda: specify the cuda device to use, default `0`





## Run EV-Gait-IMG

- generate the image-like representation

  ```
  cd EV-Gait-IMG
  python make_hdf5.py
  ```

- Download the pretrained model to the `trained_model` folder:

  https://pan.baidu.com/s/1xNbYUYYVPTwwjXeQABjmUw , extraction code: **g5k2** 

  we provide four well trained model for four image-like representations presented in the paper.

  - EV_Gait_IMG_four_channel.pkl
  - EV_Gait_IMG_counts_only_two_channel.pkl
  - EV_Gait_IMG_time_only_two_channel.pkl
  - EV_Gait_IMG_counts_and_time_two_channel.pkl

  

- run EV-Gait-IMG model with the pretrained model:

  ```
  python -u test_gait_cnn.py --img_type counts_only_two_channel --model_path EV_Gait_IMG_counts_only_two_channel.pkl
  ```




- train EV-Gait-IMG from scratch:

  ```
  nohup python -u train_gait_cnn.py --epoch 100 --cuda 0 > train_gait_cnn.log 2>&1 &
  ```

  > parameters of **train_3d_graph.py**
  >
  > - --batch_size: default `128`
  > - --epoch: number of iterations, default `50`
  > - --cuda: specify the cuda device to use, default `0`
  > - --img_type: specify the type of image-like representation to train the cnn. Four options are provided according to the paper.
  >   - `four_channel` : All four channels are considered, which is the original setup of the image-like representation
  >   - `counts_only_two_channel` : Only the two channels accommodating the counts of positive or negative events are kept.
  >   - `time_only_two_channel` : Only the two channels holding temporal characteristics are kept.
  >   - `counts_and_time_two_channel` : The polarity of the events is removed.

