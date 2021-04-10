This is the code for the paper **Event-Stream Representation for Human Gaits Identification Using Deep Neural Networks** by Yanxiang Wang, [Xian Zhang](https://blog.zhangxiann.com/), [Yiran Shen*](http://yiranshen.academic.site/), Bowen Du, Guangrong Zhao, Lizhen Cui Cui Lizhen, Hongkai Wen.

The paper can be found [here](http://academic0202101180gpyi.images.academic.site/eventstream%20representation%20early%20access%20version.pdf).

In this paper, We propose new event-based gait recognition approaches basing on two different representations of the event-stream, i.e., graph and image-like representations, and use Graph-based Convolutional Network (GCN) and Convolutional Neural Networks (CNN) respectively to recognize gait from the event-streams. The two approaches are termed as **EV-Gait-3DGraph** and **EV-Gait-IMG**. To evaluate the performance of the proposed approaches, we collect two event-based gait datasets, one from real-world experiments and the other by converting the publicly available RGB gait recognition benchmark CASIA-B.

If you use any of this code or data, please cite the following publication:

> @article{wang2021event,
>   title={Event-Stream Representation for Human Gaits Identification Using Deep Neural Networks},
>   author={Wang, Yanxiang and Zhang, Xian and Shen, Yiran and Du, Bowen and Zhao, Guangrong and Lizhen, Lizhen Cui Cui and Wen, Hongkai},
>   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
>   year={2021},
>   publisher={IEEE}
> }



# Requirements

- Python 3.x
- Conda
- cuda
- Matlab (with **Computer Vision Toolbox** and **Image Processing Toolbox** for nonuniform grid downsample)



# Installation

- First set up an [Anaconda](https://www.anaconda.com/) environment:

  ```
  conda create -n gait python=3.7  
  conda activate gait
  ```

  

- Then clone the repository and install the dependencies with pip:

  ```
  git clone ***
  cd TPAMI_Gait_Identification
  pip install -r requirements.txt
  ```

  



## Data

We use both data collected in real-world experiments(called **DVS128-Gait**) and converted from publicly available RGB gait databases(called **EV-CASIA-B**).


### DVS128-Gait

we use a [DVS128 Dynamic Vision Sensor](https://inivation.com/support/hardware/dvs128/) from iniVation operating at 128*128 pixel resolution.

we collect two dataset: **DVS128-Gait-Day** and **DVS128-Gait-Night**, which were collected under day and night lighting condition respectively.

For each lighting condition, we recruited  20 volunteers to contribute their data in two experiment sessions spanning over a few days. In each session, the participants were asked to repeat walking in front of the DVS128 sensor for 100 times.

- **DVS128-Gait-Day**: [https://drive.google.com/file/d/1i5znP-ozea-r8svMV9mLIJXblbJZqCBL/view?usp=sharing](https://drive.google.com/file/d/1i5znP-ozea-r8svMV9mLIJXblbJZqCBL/view?usp=sharing)
- **DVS128-Gait-Night**: 



## Run EV-Gait-3DGraph

- download **DVS128-Gait-Day** dataset to the `data` folder, you will get **DVS128-Gait-Day.zip**, and unzip it in that folder.


- event downsample using matlab:

  > 1. open Matlab
  >
  > 2. go to `matlab_downsample`
  >
  > 3. execute `main.m`

- generate graph representation for event:

  ```
  cd generate_graph
  python mat2graph.py
  ```

  

- Download the pretrained model:

  ```
  
  ```




- run gait recognition with the pretrained model:

  ```
  
  ```

- train from scratch:

  ```
  cd EV-Gait-3DGraph
  python train_3d_graph.py
  ```



## Run EV-Gait-IMG