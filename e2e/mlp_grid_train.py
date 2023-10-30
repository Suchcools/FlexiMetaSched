# 使用scikit-learn库中的GridSearchCV对MLP进行自动调参
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from dataloader import ML_Dataset
import joblib
import warnings
warnings.filterwarnings("ignore")
from utils import parse_opts
#加载训练数据集
opt = parse_opts()
path = opt.exact_solution
train_dataset = ML_Dataset('train',path)
test_dataset = ML_Dataset('test',path)


X_train = train_dataset.x
X_test = test_dataset.x
y_train = train_dataset.y
y_test = test_dataset.y

param_grid = {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
              'activation': ['tanh', 'relu'],
              'solver': ['sgd', 'adam'],
              'alpha': [0.0001, 0.05],
              'learning_rate': ['constant', 'adaptive']}
grid_search = GridSearchCV(MLPRegressor(max_iter=5), param_grid, cv=5) #1000
grid_search.fit(X_train, y_train)
#保存最佳模型
joblib.dump(grid_search, opt.mlp_model)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))