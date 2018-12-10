# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:17:27 2017

@author: XuFeng
"""
import numpy as np
import pandas as pd
from scipy import optimize
import time  
import scipy.io as sio  
class MUMTMC(object):
    def __init__(self,N,M,rseed):
        #users and tasks
        self.N,self.M=N,M
        #dataframe's index and columns
        self.users=['user%d'%(i+1) for i in range(N)]
        self.DIN=['DIN%d'%(i+1) for i in range(M)]
        self.DOUT=['DOUT%d'%(i+1) for i in range(M)]
        self.Task=['Task%d'%(i+1) for i in range(M)]
        #original dataframe:datain and dataout
        np.random.seed(rseed)
        self.Datain=pd.DataFrame(np.random.randint(10,31,size=(N,M)),index=self.users,columns=self.DIN)
        self.Dataout=pd.DataFrame(np.random.randint(0,1,size=(N,M)),index=self.users,columns=self.DOUT)
        self.Data = pd.concat([self.Datain, self.Dataout], axis=1,join_axes=[self.Datain.index])
        #the number of action and features
        self.n_actions = (N*M+N*2)*2
        self.n_features = N*M+N*2
        #fixed parameters
        self.APP,self.fc,self.p,self.a,self.et=1900.,10.*10**9,1,1.5*10**-7,1.42*10**-7                      
        self.El,self.Tl,self.CUL=3.25*10**-7,4.75*10**-7,150/8
        #self.APP,self.fc,self.p,self.a,self.et=25.,3.7*10**9,1,1*10**-8,1.*10**-9                      
        #self.El,self.Tl,self.CUL=1.21*10**-8,8.93*10**-9,100/8                                     
        #addional parameters
        self.Data['El'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20*self.El for i in range(N)], index=self.users)
        self.Data['et'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20*self.et for i in range(N)], index=self.users)
        #self.Data['er'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DOUT1':'DOUT%d'%M]*8*2**20*self.er for i in range(N)], index=self.users)
        self.Data['d'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20 for i in range(N)], index=self.users)
        #for i in range(N):
        #    self.Data.loc['user%d'%(i+1),'er'].columns=self.DIN
        self.Data['EC'] = self.Data.loc[:,'d']*self.a+self.Data.loc[:,'et']
        self.Data['TL'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20*self.Tl for i in range(N)], index=self.users)
        #self.Data['Tac'] = pd.Series([(self.Data.loc['user%d'%(i+1),'DOUT1':'DOUT%d'%M].tolist()+self.Data.loc['user%d'%(i+1),'DIN1':'DIN%d'%M])*8*2**20/self.Rac for i in range(N)], index=self.users)
        self.Data['Tc'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*self.APP*8*2**20/self.fc for i in range(N)], index=self.users)
        #the dataframe of offloading dicision and rate adjustment
        self.X=pd.DataFrame(np.random.randint(0,2,size=(N,M)),index=self.users,columns=self.Task)
        self.C=pd.DataFrame(np.random.rand(N,2),index=self.users,columns=['cu','cd'])
        self.XC = pd.concat([self.X, self.C], axis=1,join_axes=[self.Datain.index])
        for i in range(self.N):
            self.XC.loc['user%d'%(i+1),['cu','cd']]=[100/(8*self.N),100/(8*self.N)]


    def MUSUM_h(self,ho,M):
        
        self.Data.iloc[0,0:3]=ho[0:3]
        self.Data.iloc[1,0:3]=ho[3:6]
        self.Data.iloc[2,0:3]=ho[6:9]
        
        
        self.Data['El'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20*self.El for i in range(self.N)], index=self.users)
        self.Data['et'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20*self.et for i in range(self.N)], index=self.users)
        #self.Data['er'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DOUT1':'DOUT%d'%self.M]*8*2**20*self.er for i in range(self.N)], index=self.users)
        self.Data['d'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20 for i in range(self.N)], index=self.users)
        #for i in range(self.N):
        #    self.Data.loc['user%d'%(i+1),'er'].columns=self.DIN
        self.Data['EC'] = self.Data.loc[:,'d']*self.a+self.Data.loc[:,'et']
        self.Data['TL'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20*self.Tl for i in range(self.N)], index=self.users)
        #self.Data['Tac'] = pd.Series([(self.Data.loc['user%d'%(i+1),'DOUT1':'DOUT%d'%self.M].tolist()+self.Data.loc['user%d'%(i+1),'DIN1':'DIN%d'%self.M])*8*2**20/self.Rac for i in range(self.N)], index=self.users)
        self.Data['Tc'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*self.APP*8*2**20/self.fc for i in range(self.N)], index=self.users)
        
        self.XC.iloc[0,0:3]=M[:3]
        self.XC.iloc[1,0:3]=M[3:6]
        self.XC.iloc[2,0:3]=M[6:9]
        #start_time=time.time()
        SUM=0
        #Xi=np.random.randint(0,2,size=(self.N))
        for i in range(self.N):
            #if np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M]).any()==0:
            #    Xi[i]=0
            #else:
            #    Xi[i]=1
            sum1=(self.Data.loc['user%d'%(i+1),'El']*(1-np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M]))+self.Data.loc['user%d'%(i+1),'EC']*np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M])).iloc[0,0:].sum()
            temp1=(self.Data.loc['user%d'%(i+1),'TL']*(1-np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M]))).iloc[0,0:].sum()
            temp2=((self.Data.loc['user%d'%(i+1),'DIN1':'DIN%d'%self.M]/self.XC.loc['user%d'%(i+1),'cu']+self.Data.loc['user%d'%(i+1),'Tc'])*np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M])).iloc[0,0:].sum()
            SUM+=sum1+self.p*max(temp1,temp2)
        #end_time=time.time()
        #print('time cost:',end_time-start_time)
        return SUM

    #for test
if __name__ == "__main__":
    env = MUMTMC(3,3,1)
    #env.reset()
    #s,r,d=env.step(2)
    #h=np.array([])
    #for i in range(env.N):
    #    h=np.hstack((h,np.array(env.Data.loc['user%d'%(i+1),'DIN1':'DIN%d'%env.M])))
    #for i in range(env.N):
    #    h=np.hstack((h,np.array(env.Data.loc['user%d'%(i+1),'DOUT1':'DOUT%d'%env.M])))
    #print(h)
    dataout = sio.loadmat('MUMT_data_3x3(new)')['dataout']
    gain = sio.loadmat('MUMT_data_3x3(new)')['gain_min']
    
    h1=dataout[2,:]

    m=[1, 1, 1, 0, 0, 0, 1, 1, 1]
    #print(env.XC)
    #print(env.XC.iloc[4,4])
    #print(env.XC)
    print(h1)

    #print(env.Data)
    #print('min:',env.MUSUM(m))
    print('min:',env.MUSUM_h(h1,m))
    #print(gain[0][550])
    #print("----------------------------------------------rate of progress:%0.2f"%(120/20000))
    #print('Optimize_min:',env.Optimize_MUSUM(m))
    #print(env.XC)
    #print(env.XC)
    #print(env.Data.iloc[0,0])
    