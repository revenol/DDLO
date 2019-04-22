# DDLO
*Distributed Deep Learning-based Offloading for Mobile Edge Computing Networks*

Python code to reproduce our works on Deep Learning-based Offloading for Mobile-Edge Computing Networks [1], where multiple parallel Deep Neural Networks (DNNs) are used to efficiently generate near-optimal binary offloading decisions. This project includes:

- [memory.py](memory.py): the DNN structure for DDLO, inclduing training structure and test structure

- [data](./data): all data are stored in this subdirectory, includes:

  - **MUMT_data_3X3.mat**: training and testing data sets, where 3X3 means that the user number is 3, and each has 3 tasks.

- [main.py](main.py): run this file, inclduing setting system parameters

- [MUMT.py](MUMT.py): compute system utility Q, provided with the size of all tasks and offloading decision

## Cite this work

1. Liang Huang, Xu Feng, Anqi Feng, Yupin Huang, and Li Ping Qian, "[Distributed Deep Learning-based Oﬄoading for Mobile Edge Computing Networks](https://doi.org/10.1007/s11036-018-1177-x)," in Mobile Networks and Applications, 2018, DOI:10.1007/s11036-018-1177-x.

## Required packages

- Tensorflow

- numpy

- scipy

## How the code works

run the file, [main.py](main.py)

## Contacts

If you have any questions related to the codes, please feel free to contact *Liang Huang* (lianghuang AT zjut.edu.cn)

## Related works

For much cleaner and well-commented source codes, please refer to our recent [DROO](https://github.com/revenol/DROO) project:
1. Liang Huang, Suzhi Bi, and Ying-jun Angela Zhang, **Deep Reinforcement Learning for Online Computation Offloading in Wireless Powered Mobile-Edge Computing Networks**, on [arxiv:1808.01977](https://arxiv.org/abs/1808.01977).
