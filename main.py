#  #################################################################
#  Deep Q-learning for Wireless-powered Mobile Edge Computing.
#
#  This file contains the main code to train and test the DNN. It loads the training samples saved in ./data/data_#.mat, splits the samples into three parts (training, validation, and testing data constitutes 60%, 20% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. THere are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#  Output:
#    - Training Time: the time cost to train 18,000 independent data samples
#    - Testing Time: the time cost to compute predicted 6,000 computing mode
#    - Test Accuracy: the accuracy of the predicted mode selection. Please note that the mode selection accuracy is different from computation rate accuracy, since two different computing modes may leads to similar weighted sum computation rates. From our experience, the accuracy of weighted sum computation rate (evaluated as DNN/CD) is higher than the accuracy of computing mode selection.
#    - ./data/weights_biases.mat: parameters of the trained DNN, which are used to re-produce this trained DNN in MATLAB.
#    - ./data/Prediction_#.mat
#    Besides the test data samples, it also includes the predicted mode selection. Given DNN-predicted mode selection, the corresponding optimal weighted sum computation rate can be computed by solving (P1) in [1], which achieves over 99.9% of the CD method [2].
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       DNN-predicted mode selection    |    output_mode_pred   |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Shengli Zhang, and Ying-jun Angela Zhang, Deep Neural Network for Computation Rate Maximization in Wireless Powered Mobile-Edge Computing Systems, submitted to IEEE Wireless Communications Letters.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” submitted for publication, available on-line at arxiv.org/abs/1708.08810.
#
# version 1.0 -- January 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import MUMTMC_env as MU
from memory import MemoryDNN
import time

def plot_gain(gain_his,name=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    gain_array = np.asarray(gain_his)
    df = pd.DataFrame(gain_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
    rolling_intv = 60
    df_roll=df.rolling(rolling_intv, min_periods=1).mean()
    if name != None:
        sio.savemat('./data/MUMT(%s)'%name,{'ratio':gain_his})

    plt.plot(np.arange(len(gain_array))+1, df_roll, 'b')
    plt.fill_between(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Gain ratio')
    plt.xlabel('learning steps')
    plt.show()

def save_to_txt(gain_his, file_path):
    with open(file_path, 'w') as f:
        for gain in gain_his:
            f.write("%s \n" % gain)


def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x));
    return x;
if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
    '''
    for j in range(1):
        N = 20000                     # number of channel
        net_num = 3                   # number of DNNs
        WD_num = 3                    # number of WDs in the MERCHANTABILITY
        task_num = 3                  # number of tasks per WD

        # Load data
        dataout = sio.loadmat('./data/MUMT_data_3x3')['dataout']
        gain = sio.loadmat('./data/MUMT_data_3x3')['gain_min']

        # generate the train and test data sample index
        # data are splitted as 80:20
        # training data are randomly sampled with duplication if N > total data size
        split_idx = int(.8* len(dataout))
        num_test = min(len(dataout) - split_idx, N - int(.8* N)) # training data size

        mem = MemoryDNN(net = [WD_num*task_num, 120, 80, WD_num*task_num],net_num=net_num,
                        learning_rate = 0.01,
                        training_interval=10,
                        batch_size=128,
                        memory_size=1024
                        )

        start_time=time.time()

        gain_his = []
        gain_his_ratio = []
        knm_idx_his = []
        m_li=[]
        env = MU.MUMTMC(3,3,1)
        for i in range(N):
            if i % (N//100) == 0:
               print("----------------------------------------------rate of progress:%0.2f"%(i/N))
            if i < N - num_test:
                #training
                i_idx = i % split_idx
            else:
                # test
                i_idx = i - N + num_test + split_idx
            h1=dataout[i_idx,:]
            #pretreatment，for better train
            h=h1*10-200
            #produce offloading decision
            m_list = mem.decode(h)
            m_li.append(m_list)
            r_list = []
            for m in m_list:
                r_list.append(env.MUSUM_h(h1,m))
            # memorize the largest reward
            if i>=512:
            # record the index of largest reward
                gain_his.append(np.min(r_list))
                knm_idx_his.append(np.argmin(r_list))
                gain_his_ratio.append(gain[0][i_idx]/gain_his[-1])
                #print(m_list[np.argmin(r_list)],gain[0][i_idx]/gain_his[-1])
            # encode the mode with largest reward
            else:
                #Start learning when you have at least 512 memories (half of the memory size) in your memory bank
                #this is not necessary. But it leads to better convergence performance
                if i==511:
                    print('-------------------------------------------------')
                    print('----------------start to learn-------------------')
            mem.encode(h, m_list[np.argmin(r_list)])

        total_time=time.time()-start_time
        print('time_cost:%s'%total_time)
        print('average time per channel:%s'%(total_time/N))
        print("gain/max ratio: ", sum(gain_his_ratio[-num_test: -1])/num_test)
        print("The number of net: ", net_num)

        mem.plot_cost()
        plot_gain(gain_his_ratio,name='test')
