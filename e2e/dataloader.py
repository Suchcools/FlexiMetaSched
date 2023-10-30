import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils import data
import numpy as np
from utils import parse_opts
import warnings
warnings.filterwarnings("ignore")
### 多线程 Windows关闭
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=100)

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
    rawdata=pd.concat([rawdata,rawdata.describe()])
    rawdata = rawdata.append(pd.Series(p.flatten().tolist()+[0] * (column_length-4), index=rawdata.columns), ignore_index=True)
    return rawdata.to_numpy()


def load_data(path):
    # 读取合并X和Y
    df = pd.read_csv(path)
    df['feature'] = df.path.parallel_apply(lambda x:env2vec(x.replace('./','env/')).flatten().tolist())
    df.to_feather('/home/linjw/iProject/HA-Prototypical/env/env2vec.feather')
    # df = pd.read_feather('env/env2vec.feather')
    return df

def process_data(df, mode, type = 'MAML'):
    if mode != 'test' and type != 'MLP':
        df = np.vstack((df.values, df.values))
    else:
        df = df.values
    x = np.array(df[:,-1])
    x = np.array([list(i) for i in x],dtype=np.float32)
    y = [np.array(eval(i)).flatten() for i in df[:,-3]]
    y = np.array([list(x) for x in y])
    y_index = np.arange(len(y)) % (len(y)/2 if mode != 'test' else len(y)/2)
    return x, y_index, {i: y[i] for i in range(len(y))}, y

# 定义MAML_Dataset类
class MAML_Dataset(data.Dataset):
    def __init__(self, mode, path, test_size=0.2, random_state=0, ood = True ):
        super().__init__()
        # self.sample_len = 1024
        self.df = load_data(path)
        if ood:
            split=int(0.75*len(self.df))
            self.train_df, self.test_df = self.df.iloc[:split],self.df.iloc[split:]
        else:
            self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)          
        self.x, self.y, self.label_dict, _ = process_data(self.train_df if mode != 'test' else self.test_df,mode)

    def __getitem__(self, index):
        x = np.array(self.x[index],dtype=np.float32)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)
    

# 定义ML_Dataset类
class ML_Dataset(data.Dataset):
    def __init__(self, mode, path, test_size=0.2, random_state=0):
        super().__init__()
        self.sample_len = 1024
        self.df = load_data(path)
        self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        self.id = self.test_df.iloc[:,-1]
        self.x, _ ,_ , self.y= process_data(self.train_df if mode != 'test' else self.test_df,mode, type ='MLP')

    def __getitem__(self, index):
        x = np.array(self.x[index],dtype=np.float32)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)
    
if __name__ == "__main__":
    opt = parse_opts()
    # dataset = ML_Dataset('train',path=opt.exact_solution)
    # dataset = MAML_Dataset('train',path=opt.exact_solution)
    # print(dataset[0][0].shape) 