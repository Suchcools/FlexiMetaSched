# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import pandas as pd
import numpy as np
import torch 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
random_state = 40

def env2vec(path):
    Job_pd=pd.read_excel(path,sheet_name='Job info')
    Mac_pd=pd.read_excel(path,sheet_name='Machine info')
    Pr1_pd=pd.read_excel(path,sheet_name='Process 1 Time')
    Pr2_pd=pd.read_excel(path,sheet_name='Process 2 Time')
    # 定义问题的参数
    J = len(Job_pd)
    M = len(Mac_pd)
    W = Job_pd['准备时间'].mean()  # 订单的 Task0 和 Task1 之间的准备时间
    N = Job_pd['需求量'].values  # 订单的需求量
    A = Job_pd['到达时刻'].values  # 订单的到达时刻
    D = Job_pd['交货期'].values  # 订单的交货期
    pt= np.array([Pr1_pd[['Machine_0','Machine_1']].values,Pr2_pd[['Machine_0','Machine_1']].values])# 订单的 Task0 在生产线 i 上的单位处理时长 # 订单的 Task1 在生产线 i 上的单位处理时长
    p = np.array([Mac_pd[['闲置功率','工作功率']].values[:,0],Mac_pd[['闲置功率','工作功率']].values[:,1]])   # 生产线的闲置功率和工作功率
    rawdata=pd.concat([Job_pd,pd.DataFrame(N*p[1].mean(),columns=['预计功耗']),Pr1_pd[['Machine_0','Machine_1']],Pr2_pd[['Machine_0','Machine_1']]],axis=1).iloc[:,1:]
    rawdata=pd.concat([Job_pd,pd.DataFrame(N*p[1].mean(),columns=['预计功耗']),Pr1_pd[['Machine_0','Machine_1']],Pr2_pd[['Machine_0','Machine_1']]],axis=1).iloc[:,1:]
    column_length=len(rawdata.columns)
    # 创建一个新的行，所有值均为0
    zero_row = pd.Series([0] * column_length, index=rawdata.columns)
    # 将新行附加到DataFrame末尾
    dsize=len(rawdata)
    for i in range(200-dsize):
        rawdata = rawdata.append(zero_row, ignore_index=True)
    rawdata=pd.concat([rawdata,rawdata.describe()])
    rawdata = rawdata.append(pd.Series(p.flatten().tolist()+[0] * (column_length-4), index=rawdata.columns), ignore_index=True)
    return rawdata.to_numpy()

def data_split(df, mode, select_type):
    if select_type == 1:
        # 选取四个分类作为训练测试集
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=random_state)
    elif select_type == 2:
        # 选取0和1两个类别作为训练测试集
        df_01 = df[df['label'].isin([0, 1])]
        train_df, test_df = train_test_split(df_01, test_size=0.2, stratify=df_01['label'], random_state=random_state)
    elif select_type == 3:
        # 训练集选取0和1，测试集选取1和2 #启发式方法
        train_df = df[df['label'].isin([0, 1])]
        test_df = df[df['label'].isin([1, 2])]
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    else:
        raise ValueError("Invalid 'type' value. It must be one of [1, 2, 3].")
    
    if mode == 'train':
        return train_df
    elif mode == 'test':
        return test_df
    else:
        raise ValueError("Invalid 'mode' value. It must be either 'train' or 'test'.")


class EnvDataset(data.Dataset):

    def __init__(self, mode='train', root='prototypical/dataset/data_info.csv', select_type = 1 ):
        data = pd.read_csv(root)
        data['path'] = data['path'].apply(lambda x:x.replace('../','./'))
        dataset = data_split(data, mode, select_type = select_type) ### 设置了三种模式 1 全采样四分类 2 选取1和2 两个类别 二分类混合 3 选取1和2训练集 3 4为测试集 测试泛化性能
        self.y = torch.from_numpy(dataset.label.values)
        self.x = [env2vec(x)for x in dataset.path.values]
        self.x = [torch.unsqueeze(torch.Tensor(x), 0) for x in self.x]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


dataset=EnvDataset()
for xx,yy in dataset :
    break 