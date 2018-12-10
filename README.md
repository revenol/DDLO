# DDLO
*Distributed Deep Learning-based Offloading for Mobile Edge Computing Networks*

Python code to reproduce our works on Deep Learning-based Offloading for Mobile-Edge Computing Networks [1], where multiple parallel Deep Neural Networks (DNNs) are used to efficiently generate near-optimal binary offloading decisions. This project includes:

- [memory.py](memory.py): the DNN structure for the DDLO, inclduing training structure and test structure

- [data](./data): all data are stored in this subdirectory, includes:

  - **data_#.mat**: training and testing data sets, where # = {10, 20, 30} is the user number

- [main.py](main.py): run this file, inclduing setting system parameters


## Cite this work

1. Liang Huang, Xu Feng, Anqi Feng, Yupin Huang, and Li Ping Qian, "[Distributed Deep Learning-based Oﬄoading for Mobile Edge Computing Networks](https://doi.org/10.1007/s11036-018-1177-x)," in Mobile Networks and Applications, 2018, DOI:10.1007/s11036-018-1177-x.

## Required packages

- Tensorflow

- numpy

- scipy

## How the code works

run the file, [main.py](main.py)
