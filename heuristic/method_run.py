from methods import aco, ga, sa, ts
from methods import common

# 输入环境
path = 'scene_env/case1/0.xlsx'
for i in range(1):
    print(f'Round { i +1 }')
    print('  SA : ', sa(common.read_env(path)))
    # print('  GA : ', ga(common.read_env(path)))
    # print('  TS : ', ts(common.read_env(path)))
    # print('  ACO : ', aco(common.read_env(path)))
