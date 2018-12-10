# DDLO
*Distributed Deep Learning-based Offloading for Mobile Edge Computing Networks*

Python code to reproduce our works on Deep Learning-based Offloading for Mobile-Edge Computing Networks [1], where multiple parallel Deep Neural Networks (DNNs) are used to efficiently generate near-optimal binary offloading decisions. This project includes:

- [memory.py](memory.py): the DNN structure for the WPMEC, inclduing training structure and test structure

- [data](./data): all data are stored in this subdirectory, includes:

  - **MUMT_data_3X3.mat**: training and testing data sets, where 3X3 means that the user number is 3, and each has 3 tasks.

- [main.py](main.py): run this file, inclduing setting system parameters

- [MEC_env.py](MEC_env.py): calculation of the underlying energy consumption, input the size of all tasks and offloading decision, and output the calculation results

## Cite this work

1. Liang Huang, Xu Feng, Anqi Feng, Yupin Huang, and Li Ping Qian, "[Distributed Deep Learning-based Oﬄoading for Mobile Edge Computing Networks](https://doi.org/10.1007/s11036-018-1177-x)," in Mobile Networks and Applications, 2018, DOI:10.1007/s11036-018-1177-x.

## Required packages

- Tensorflow

- numpy

- scipy

## How the code works

run the file, [main.py](main.py)