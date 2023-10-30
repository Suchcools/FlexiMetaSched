import numpy as np
import pandas as pd
from pandarallel import pandarallel
import warnings
warnings.filterwarnings("ignore") 
from BbCalculate import BnB

import time
start_time = time.time() 
pandarallel.initialize(nb_workers=160,progress_bar=True)
# data = np.load('1.npz',allow_pickle=True)
env_info = pd.read_csv('./env/env_info.csv')
env_info.head().apply(lambda x:BnB('env/'+x.path.replace('./',''),f'bnb/result/{x["class"]}_{x["excel_num"]}'),axis=1)
end_time = time.time()    # 程序结束时间
run_time = end_time - start_time    # 程序的运行时间，单位为秒
print(run_time)