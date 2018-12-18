#  #################################################################
#  This file compute the system utility Q, provided with the size of all tasks and offloading decision
#
# version 1.0 -- January 2018. Written by Xu Feng (xfeng_zjut AT 163.com)
#  #################################################################
import numpy as np
import pandas as pd
class MUMT(object):
    def __init__(self,N,M,rand_seed=1):
        #users and tasks
        self.N,self.M=N,M
        #dataframe's index and columns
        self.users=['user%d'%(i+1) for i in range(N)]
        self.DIN=['DIN%d'%(i+1) for i in range(M)]
        self.DOUT=['DOUT%d'%(i+1) for i in range(M)]
        self.Task=['Task%d'%(i+1) for i in range(M)]
        #original dataframe:datain and dataout
        np.random.seed(rand_seed)
        self.Datain=pd.DataFrame(np.random.randint(10,31,size=(N,M)),index=self.users,columns=self.DIN)
        self.Dataout=pd.DataFrame(np.random.randint(0,1,size=(N,M)),index=self.users,columns=self.DOUT)
        self.Data = pd.concat([self.Datain, self.Dataout], axis=1,join_axes=[self.Datain.index])
        #fixed parameters
        self.APP,self.fc,self.p,self.a,self.et=1900.,10.*10**9,1,1.5*10**-7,1.42*10**-7
        self.El,self.Tl,self.CUL=3.25*10**-7,4.75*10**-7,150/8
        #addional parameters
        self.Data['El'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20*self.El for i in range(N)], index=self.users)
        #the local energy consumption
        self.Data['et'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20*self.et for i in range(N)], index=self.users)
        #transmission energy consumption
        self.Data['d'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20 for i in range(N)], index=self.users)
        self.Data['EC'] = self.Data.loc[:,'d']*self.a+self.Data.loc[:,'et']
        #energy consumption of the edge server
        self.Data['TL'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*8*2**20*self.Tl for i in range(N)], index=self.users)
        #local time delay
        self.Data['Tc'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%M]*self.APP*8*2**20/self.fc for i in range(N)], index=self.users)
        #The edge server delay
        self.X=pd.DataFrame(np.random.randint(0,2,size=(N,M)),index=self.users,columns=self.Task)
        self.C=pd.DataFrame(np.random.rand(N,2),index=self.users,columns=['cu','cd'])
        self.XC = pd.concat([self.X, self.C], axis=1,join_axes=[self.Datain.index])
        for i in range(self.N):
            self.XC.loc['user%d'%(i+1),['cu','cd']]=[100/(8*self.N),100/(8*self.N)]


    def compute_Q(self,task_size,M):
        #Provide task_size and offloading decision, and compute the result.
        self.Data.iloc[0,0:3]=task_size[0:3]
        self.Data.iloc[1,0:3]=task_size[3:6]
        self.Data.iloc[2,0:3]=task_size[6:9]

        self.Data['El'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20*self.El for i in range(self.N)], index=self.users)
        #the local energy consumption
        self.Data['et'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20*self.et for i in range(self.N)], index=self.users)
        #transmission energy consumption
        self.Data['d'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20 for i in range(self.N)], index=self.users)
        self.Data['EC'] = self.Data.loc[:,'d']*self.a+self.Data.loc[:,'et']
        #energy consumption of the edge server
        self.Data['TL'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*8*2**20*self.Tl for i in range(self.N)], index=self.users)
        #local time delay
        self.Data['Tc'] = pd.Series([self.Data.loc[['user%d'%(i+1)],'DIN1':'DIN%d'%self.M]*self.APP*8*2**20/self.fc for i in range(self.N)], index=self.users)
        #The edge server delay
        self.XC.iloc[0,0:3]=M[:3]
        self.XC.iloc[1,0:3]=M[3:6]
        self.XC.iloc[2,0:3]=M[6:9]
        SUM=0
        for i in range(self.N):
            sum1=(self.Data.loc['user%d'%(i+1),'El']*(1-np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M]))+self.Data.loc['user%d'%(i+1),'EC']*np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M])).iloc[0,0:].sum()
            temp1=(self.Data.loc['user%d'%(i+1),'TL']*(1-np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M]))).iloc[0,0:].sum()
            temp2=((self.Data.loc['user%d'%(i+1),'DIN1':'DIN%d'%self.M]/self.XC.loc['user%d'%(i+1),'cu']+self.Data.loc['user%d'%(i+1),'Tc'])*np.array(self.XC.loc['user%d'%(i+1),'Task1':'Task%d'%self.M])).iloc[0,0:].sum()
            SUM+=sum1+self.p*max(temp1,temp2)
            #Integrate energy consumption and time delay
        return SUM
