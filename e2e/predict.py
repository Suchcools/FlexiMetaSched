import joblib
import pandas as pd
import numpy as np
from dataloader import ML_Dataset
from dataloader import MAML_Dataset
from model import MLP
import torch
import warnings
warnings.filterwarnings("ignore")
from utils import parse_opts
import sys
sys.path.append('heuristic')
from methods.common import parse_env,ruler,fitness


def test_model_performance(path, load_path, save_path, model_type, solution):
    """
    Test the performance of a trained model on a test dataset.
    
    Args:
        path (str): The path to the dataset.
        load_path (str): The path to the trained model's weights.
        save_path (str): The path to save the performance dataframe as an Excel file.
        model_type (str): The type of the model ('MLP' or 'grid_search').
        
    Returns:
        performance_df (pandas.DataFrame): A dataframe containing the performance metrics.
    """


    # Load the test dataset and object results

    if model_type == 'MAML':
        test_dataset = MAML_Dataset(mode='test', path=path)
        test_df = test_dataset.test_df
        test_df.index=range(len(test_df))
        # Load the trained model and predict the output for the test dataset
        models = MLP(756, 256, 450, dropout=0.1)
        models.load_state_dict(torch.load(load_path))
        models.eval()
        feature = test_df.iloc[:,-1].values
        feature = np.array([list(x) for x in feature],dtype=np.float32)
        feature = torch.from_numpy(feature).float()
        # ### test
        import learn2learn as l2l
        maml = l2l.algorithms.MAML(models, lr=1e-10)
        y_data_list =  maml(feature).detach().cpu().numpy()
        # product_list = formate(y_data_list,test_df.bob,75)
        # ### test
        y_data_list = models(feature).detach().cpu().numpy()
        product_list = [ruler(x.reshape(150,3),75) for x in y_data_list]  # 产线还原
    elif model_type == 'MLP':
        test_dataset = ML_Dataset(mode='test', path=path)
        test_df = test_dataset.test_df
        test_df.index=range(len(test_df))
        # Load the trained model and predict the output for the test dataset
        grid_search_mlp = joblib.load(load_path)
        feature = test_df.iloc[:,-1].values
        feature = np.array([list(x) for x in feature],dtype=np.float32)
        y_data_list = grid_search_mlp.predict(torch.from_numpy(feature).float())
        product_list = [ruler(x.reshape(150,3),75) for x in y_data_list]  # 产线还原

    # Predict the time taken for each file using the output from the model


    predicted_time_list = []
    for i, file_name in test_df.iterrows():
        excel_name = file_name['path'].replace('./','env/')
        order = product_list[i]
        J, M, A, D, N, pt, p, W = parse_env(excel_name)
        # -fitness(order, J, M, A, D, N, pt, p, W)
        predicted_time_list.append(-fitness(order, J, M, A, D, N, pt, p, W))
        # predicted_time_list.append(max(calculate_cost(order,J, M, A, D, N, pt, p, W)[1].values()))

    # Convert the object results for the test dataset to a list
    true_time_list = test_df['fitness'].values.tolist()


    # Calculate the performance metrics
    error_list = []
    percent_error_list = []
    for i in range(len(true_time_list)):
        true_time_list[i] = float(true_time_list[i])
        predicted_time_list[i] = float(predicted_time_list[i])
        error_list.append(true_time_list[i] - predicted_time_list[i])
        percent_error_list.append((true_time_list[i] - predicted_time_list[i]) / true_time_list[i])
     
    pd.DataFrame([predicted_time_list,true_time_list],index=['Predict','GroundTruth']).T.to_csv(f'{opt.predict_output}/{model_type}_label.csv',index=False)
    # Create a dataframe with the performance metrics and write it to an Excel file
    performance_df = pd.DataFrame(columns=['average error', 'average percent error', 'predicted time average', 'predicted time max', 'predicted time min'])
    performance_df.loc[0] = [sum(error_list)/len(error_list), sum(percent_error_list)/len(percent_error_list), sum(predicted_time_list)/len(predicted_time_list), max(predicted_time_list), min(predicted_time_list)]
    performance_df.to_excel(save_path)

    print(performance_df)
if __name__ == "__main__":
    opt = parse_opts()
    test_model_performance(opt.exact_solution, opt.maml_model, opt.predict_output+'/rs_df_4.xlsx', 'MAML' , opt.exact_solution)
    # test_model_performance(opt.exact_solution, opt.mlp_model, opt.predict_output+'/rs_df_b4.xlsx', 'MLP', opt.exact_solution)