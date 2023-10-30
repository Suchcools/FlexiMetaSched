import pandas as pd
from methods import ga_ft
from methods import common
import numpy as np
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=100,progress_bar=True)
root='scene_env/'
rawdata=pd.read_csv('./scene_env/env_info.csv')
rawdata.path=rawdata.path.apply(lambda x:x.replace('./',root))
times = 10
rawdata=rawdata.iloc[:1800]
rawdata=rawdata.groupby('class').head(30)

def get_mean_quick(path):
    env = common.read_env(path)
    sa_lst=[]
    ga_lst=[]
    aco_lst=[]
    ts_lst=[]
    for i in range(times):
        sa_lst.append(ga_ft(env,0.05))
        ga_lst.append(ga_ft(env,0.1))
        aco_lst.append(ga_ft(env,0.15))
        ts_lst.append(ga_ft(env,0.2))
    return sa_lst,ga_lst,aco_lst,ts_lst

temp=rawdata.path.parallel_apply(lambda x: get_mean_quick(x))
a1=pd.DataFrame([x[0] for x in temp])
a2=pd.DataFrame([x[1] for x in temp])
a3=pd.DataFrame([x[2] for x in temp])
a4=pd.DataFrame([x[3] for x in temp])
pd.concat([a1,a2,a3,a4],axis=1,ignore_index=True).to_csv(f'result/result_prob.csv',index=False)