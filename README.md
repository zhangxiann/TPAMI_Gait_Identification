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

  

- Then clone the repository and install the dependencies with pip:

  ```
  git clone https://github.com/zhangxiann/TPAMI_Gait_Identification.git
  cd TPAMI_Gait_Identification
  pip install -r requirements.txt
  ```

- Install [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) according to their documentation respectively.

  - PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
  - PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)



## Data

We use both data collected in real-world experiments(called **DVS128-Gait**) and converted from publicly available RGB gait databases(called **EV-CASIA-B**).


### DVS128-Gait

we use a [DVS128 Dynamic Vision Sensor](https://inivation.com/support/hardware/dvs128/) from iniVation operating at 128*128 pixel resolution.

we collect two dataset: **DVS128-Gait-Day** and **DVS128-Gait-Night**, which were collected under day and night lighting condition respectively.

For each lighting condition, we recruited  20 volunteers to contribute their data in two experiment sessions spanning over a few days. In each session, the participants were asked to repeat walking in front of the DVS128 sensor for 100 times.

- **DVS128-Gait-Day**: [https://drive.google.com/file/d/1i5znP-ozea-r8svMV9mLIJXblbJZqCBL/view?usp=sharing](https://drive.google.com/file/d/1i5znP-ozea-r8svMV9mLIJXblbJZqCBL/view?usp=sharing)
- **DVS128-Gait-Night**: https://pan.baidu.com/s/1txWR75DaAOyva6oUOJ4Kbg , extraction code: **iypf**



## Run EV-Gait-3DGraph

- download **DVS128-Gait-Day** dataset to the `data` folder, you will get **DVS128-Gait-Day.zip**, and unzip it in the `data` folder.


- event downsample using matlab:

  > 1. open Matlab
  >
  > 2. go to `matlab_downsample`
  >
  > 3. run `main.m`

- generate graph representation for event:

  ```
  cd generate_graph
  python mat2graph.py
  ```

  
  
- Download the pretrained model to the `trained_model` folder:

  ```
  
  ```




- run gait recognition with the pretrained model:

  ```
  python test_3d_graph.py --model_path EV_Gait_3DGraph(94.25).pkl
  ```

- train from scratch:

  ```
  nohup python -u EV-Gait-3DGraph/train_3d_graph.py --epoch 80 --cuda 0 > train_3d_graph.log 2>&1 
  &
  ```
  
  the traning log would be created at `log/train.log`.





## Run EV-Gait-IMG



```
nohup python -u train_gait_cnn.py --epoch 100 --cuda 0 > train_gait_cnn.log 2>&1 &
```

