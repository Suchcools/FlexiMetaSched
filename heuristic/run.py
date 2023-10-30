import pandas as pd
from methods import aco, ga, sa, ts
from methods import common
import numpy as np
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=200,progress_bar=True)
import random
random.seed(42)
root='env/'
rawdata=pd.read_csv('./env/env_info.csv')
rawdata.path=rawdata.path.apply(lambda x:x.replace('./',root))
times = 5
def get_mean_quick(path):
    env = common.read_env(path)
    sa_lst=[]
    ga_lst=[]
    aco_lst=[]
    ts_lst=[]
    for i in range(times):
        sa_lst.append(sa(env))
        ga_lst.append(ga(env))
        aco_lst.append(aco(env))
        ts_lst.append(ts(env))
    return sa_lst,ga_lst,aco_lst,ts_lst

temp=rawdata.path.parallel_apply(lambda x: get_mean_quick(x))
# temp=rawdata.groupby('class').head(50).path.parallel_apply(lambda x: get_mean_quick(x))
a1=pd.DataFrame([x[0] for x in temp])
a2=pd.DataFrame([x[1] for x in temp])
a3=pd.DataFrame([x[2] for x in temp])
a4=pd.DataFrame([x[3] for x in temp])
pd.concat([a1,a2,a3,a4],axis=1,ignore_index=True).to_csv(f'heuristic/result/result_method.csv',index=False)