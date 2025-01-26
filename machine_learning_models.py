import itertools
from typing import *
import numpy as np
import pandas as pd
import os
import random
import math

# Tensorflow/Keras
import keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Lambda, Input
from tensorflow.python.ops import math_ops

# Sklearn
from sklearn import neighbors, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from ml_utils import tanimoto_from_sparse #, untransform_data

import warnings

#torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

#XGboost
import xgboost as xgb

###############################ADDTIONAL FOR SERA#############################
#import SERA
import torch
from torch.utils.data import DataLoader
from sera_opt_proto import phi,sera_pt,SeraCriterion
from sklearn.metrics import make_scorer
from xgboost_sera import sera_loss

#%%

os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"


def SERA_opt(y_true,y_pred,ph,device="cpu"):
    phi_labels=SERA_phi_control(y_true,ph)
    sera=sera_pt(torch.Tensor(y_true),torch.Tensor(y_pred),phi_labels,device=device)
    return float(np.array(sera))

def SERA_phi_control(y,ph,out_dtype_np=np.float32):
    #y = np.float64(np.array(y))
    #phi_values = phi(pd.Series(y.flatten()), phi_parms=ph)
    phi_values=phi(pd.Series(y.flatten(),dtype="float64"), phi_parms=ph)
    return torch.Tensor(np.array(phi_values, dtype=out_dtype_np))

#%%


sera_relevances=[5,6,7,8]
###################################################################################

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

class Dataset:
    def __init__(self, features: np.array, labels: np.array):
        self.features = features
        self.labels = labels
        self._add_instances = set()

    def add_instance(self, name, values: np.array):
        self._add_instances.add(name)
        self.__dict__[name] = values

    @property
    def columns(self) -> dict:
        data_dict = {k: v for k, v in self.__dict__.items()}
        data_dict['features'] = self.features
        data_dict['labels'] = self.labels
        return data_dict

    def __len__(self):
        return self.labels.shape[0]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}
        
        subset = Dataset(self.features[idx], self.labels[idx])
        for addt_instance in self._add_instances:
            subset.add_instance(addt_instance, self.__dict__[addt_instance][idx])

        return subset

class WeightedKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, metric='minkowski', p=2):
        """
        Initialize the WeightedKNNRegressor.

        Parameters:
        - n_neighbors (int): Number of neighbors to consider.
        - metric (str): Distance metric to use. Default is 'minkowski'.
        - p (int): Parameter for Minkowski metric (e.g., p=2 for Euclidean distance).
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.neigh = None
        self.X_train = None
        self.y_train = None
        self.sample_weights = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the WeightedKNNRegressor model.

        Parameters:
        - X (array-like): Training data of shape (n_samples, n_features).
        - y (array-like): Target values of shape (n_samples,).
        - sample_weight (array-like): Weights for each sample. Default is None.
        """
        self.neigh = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, p=self.p)
        self.neigh.fit(X)
        self.X_train = X
        self.y_train = y
        self.sample_weights = sample_weight if sample_weight is not None else np.ones(len(y))
        return self

    def predict(self, X):
        """
        Predict target values for given data.

        Parameters:
        - X (array-like): Test data of shape (n_samples, n_features).

        Returns:
        - predictions (array): Predicted target values of shape (n_samples,).
        """
        distances, indices = self.neigh.kneighbors(X)

        predictions = []
        for i, neighbors in enumerate(indices):
            neighbor_distances = distances[i]
            neighbor_weights = self.sample_weights[neighbors]
            
            # Use inverse distance weighting, adjusting with sample weights
            if np.any(neighbor_distances == 0):
                # Avoid division by zero; if a point has distance 0, use its value directly
                zero_dist_idx = np.where(neighbor_distances == 0)[0][0]
                predictions.append(self.y_train[neighbors[zero_dist_idx]])
            else:
                # Compute weighted average
                weights = neighbor_weights / (neighbor_distances + 1e-9)  # Avoid division by zero
                predictions.append(np.dot(weights, self.y_train[neighbors]) / weights.sum())

        return np.array(predictions)

    def get_params(self, deep=True):
        """
        Get the parameters of the WeightedKNNRegressor as a dictionary.

        Parameters:
        - deep (bool): If True, return parameters for this estimator and contained subobjects.
        
        Returns:
        - params (dict): Dictionary of model parameters.
        """
        return {'n_neighbors': self.n_neighbors, 'metric': self.metric, 'p': self.p}

    def set_params(self, **params):
        """
        Set the parameters of the WeightedKNNRegressor.

        Parameters:
        - params (dict): Dictionary of parameters to set.
        
        Returns:
        - self: The updated estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

class MLModel:
    def __init__(self, data, ml_algorithm, metric, ph, weights=None, reg_class="regression",
                 parameters='grid', cv_fold=10, random_seed=2002):

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.metric=metric
        self.ph=ph
        self.weights=weights
        self.reg_class = reg_class
        self.cv_fold = cv_fold
        self.seed = random_seed
        self.parameters = parameters
        self.h_parameters = self.hyperparameters()
        self.model, self.cv_results = self.cross_validation()
        self.best_params = self.optimal_parameters()
        self.model = self.final_model()
        

    def hyperparameters(self):
        if self.parameters == "grid":
            if self.reg_class == "regression":
                if self.ml_algorithm == "MR":
                    return {'strategy': ['median']
                            }
                elif self.ml_algorithm == "SVR":
                    return {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 10000],
                            'kernel': [tanimoto_from_sparse],
                            }
                elif self.ml_algorithm == "RFR":
                    return {'n_estimators': [25, 100, 200],
                                'max_features': ['auto'],
                                'min_samples_split': [2, 3, 5],
                                'min_samples_leaf': [1, 2, 5],
                                }
                elif self.ml_algorithm == "kNN":
                    return {"n_neighbors": [1, 3, 5]
                            }
                elif self.ml_algorithm=="XGBoost":
                     return {'n_estimators': [10, 50, 100, 250], 
                             'learning_rate': [.001, .01, .1], 
                             'max_depth': [3, 5, 10],
                             }

            if self.reg_class == "classification":
                if self.ml_algorithm == "SVM":
                    return {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10],
                            'kernel': [tanimoto_from_sparse],
                            }
                elif self.ml_algorithm == "RFC":
                    return {'n_estimators': [25, 100, 200],
                            'max_features': ['auto'],
                            'min_samples_split': [2, 3, 5],
                            'min_samples_leaf': [1, 2, 5],
                            }


    def cross_validation(self):

        if self.reg_class == "regression":
            sera_score=make_scorer(SERA_opt,greater_is_better=False,ph=self.ph)
            metrics_dict={"MAE":"neg_mean_absolute_error","SERA":sera_score,"MSE":"neg_mean_squared_error"}
            opt_metric= metrics_dict[self.metric]
            if self.ml_algorithm == "MR":
                model = DummyRegressor()
            elif self.ml_algorithm == "SVR":
                model = SVR()
            elif self.ml_algorithm == "RFR":
                model = RandomForestRegressor(random_state=self.seed)
            elif self.ml_algorithm == "kNN":
                if self.weights is None:
                    model = neighbors.KNeighborsRegressor()
                else:
                    model = WeightedKNNRegressor()
            elif self.ml_algorithm=="XGBoost":
                loss_xgboost={"MAE":"reg:absoluteerror","SERA":sera_loss(self.ph),"MSE":"reg:squarederror"}
                loss=loss_xgboost[self.metric]
                model=xgb.XGBRegressor(random_state=self.seed, objective=loss)

        elif self.reg_class == "classification":
            opt_metric = "balanced_accuracy"
            if self.ml_algorithm == "SVM":
                model = SVC()
            elif self.ml_algorithm == "RFC":
                model = RandomForestClassifier()

        cv_results = GridSearchCV(model,
                                  param_grid=self.h_parameters,
                                  cv=self.cv_fold,
                                  scoring=opt_metric,
                                  n_jobs=-1,
                                )
        cv_results.fit(self.data.features, self.data.labels, sample_weight=self.weights)
        
        return model, cv_results

    def optimal_parameters(self):
        best_params = self.cv_results.cv_results_['params'][self.cv_results.best_index_]
        return best_params

    def final_model(self):
        model = self.model.set_params(**self.best_params)
        return model.fit(self.data.features, self.data.labels)


import copy   
class EarlyStopping_torch:
    def __init__(self, patience=50, min_delta=0, restore_best_weights=False):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False

class WeightedMSELoss(nn.Module):
    """
    Custom Weighted Mean Squared Error Loss.

    Parameters:
    - weights: Tensor of weights for each sample.
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predictions, targets, weights):
        """
        Compute the weighted MSE loss.

        Parameters:
        - predictions (Tensor): Predicted values, shape (n_samples,).
        - targets (Tensor): Ground truth values, shape (n_samples,).
        - weights (Tensor): Sample weights, shape (n_samples,).

        Returns:
        - loss (Tensor): Weighted MSE loss.
        """
        # Compute squared differences
        squared_diff = (predictions - targets) ** 2
        
        # Apply weights to the squared differences
        weighted_squared_diff = weights * squared_diff

        # Compute the mean of the weighted squared differences
        loss = torch.mean(weighted_squared_diff)
        return loss

class DNN_torch(torch.nn.Module):
    def __init__(self,layers=None, dropout_rate= 0, activation='tanh',
                 n_features=None, seed= None, reg_class="regression"):
        # Init parent
        super(DNN_torch, self).__init__()
        torch.manual_seed(seed) 
        
        self.layers=layers
        self.activation=F.tanh if activation=='tanh' else F.relu
        self.dropout=nn.Dropout(p=dropout_rate)
        

        if layers is None:
            layers = (100, 100)
        
        self.list_layers=nn.ModuleList()
        
        for i, net_nodes in enumerate(layers, 0):
            if i == 0:
                self.list_layers.append(nn.Linear(n_features,net_nodes))
            else:
                self.list_layers.append(nn.Linear(layers[i-1],net_nodes))

        if reg_class == "regression":  
                self.list_layers.append(nn.Linear(layers[len(layers)-1],1))
    
    def forward(self,x):
        
        for i,layer in enumerate(self.list_layers, 1):
            if i==len(self.list_layers):
                x=layer(x)
            else: 
                x=self.activation(layer(x))
                x=self.dropout(x)
        
        return x
            
class DNN:
    def __init__(self, data, ml_algorithm, n_features, seed,ph, metric="SERA", weights=None,
                 reg_class="regression", parameters='grid', device="cuda"):

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.ph=ph
        self.metric=metric
        self.weights=weights
        if self.weights is None:
           self.weights=np.ones(len(self.data.labels))
        self.n_features = n_features
        self.seed = seed
        self.reg_class = reg_class
        self.parameters = parameters
        self.device = device
        self.h_parameters = self.dnn_hyperparameters()
        self.cv_results = self.dnn_cross_validation()
        self.best_params = self.dnn_select_best_parameters()
        self.model = self.final_model()
        

    
    def dnn_hyperparameters(self):
        if self.parameters == "grid":

            return {
                "module__layers": [(100, 100), (250, 250), (250, 500), (500, 250), (500, 250, 100), (100, 250, 500)],
                "module__dropout_rate": [0.0, 0.25, 0.5],
                "module__activation": ['tanh'],
                "optimizer__lr": [0.1, 0.01, 0.001],
                "max_epochs": [200]
            }

        
    def dnn_train(self,features_train,labels_train,features_valid,labels_valid,
                  nn_layers,dropout_rate,activation,learning_rate,train_epochs,weights_train,weights_valid):
            loader_train = DataLoader(list(zip(features_train,labels_train,weights_train)), shuffle=True, batch_size=32)
            if features_valid!=None:
                loader_val = DataLoader(list(zip(features_valid,labels_valid,weights_valid)), shuffle=True, batch_size=32)
            
            
            if (self.device=="cuda" and not torch.cuda.is_available()) or self.device=="cpu":
                print("Running on CPU")
                device = torch.device("cpu")
            else:
                device="cuda"
           
            
            model=DNN_torch(layers=nn_layers,
                            dropout_rate=dropout_rate,
                            activation=activation,
                            n_features=self.n_features,
                            seed=self.seed)
            model=model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            loss_dict={"WMSE":WeightedMSELoss(),"MSE":torch.nn.MSELoss(),"MAE":torch.nn.L1Loss(),"SERA": SeraCriterion(self.ph,device=device)}
            criterion =loss_dict[self.metric] 
            es = EarlyStopping_torch()
             
            done =False
            history={"train_loss":[],"val_loss":[]}
              
            for epoch in range(train_epochs):
                train_loss = 0.0
                valid_loss = 0.0 
                if not done:
                    for X_batch, y_batch, w_batch in loader_train:
                            optimizer.zero_grad()
                            # Passing the node features and the connection info
                            if device=="cuda":
                                pred=model(torch.FloatTensor(X_batch).cuda()) 
                                y=torch.FloatTensor(y_batch).cuda()
                                w=torch.FloatTensor(w_batch).cuda()
                            else:
                                pred=model(torch.FloatTensor(X_batch))  
                                y=torch.FloatTensor(y_batch)
                                w=torch.FloatTensor(w_batch)
                            # Calculating the loss and gradients´
                            if self.metric=="WMSE":
                                loss = criterion(pred,y,w)
                            else:
                                loss = criterion(pred,y)
                            loss.backward() 
                            # Update using the gradients 
                            train_loss += loss.item()
                            optimizer.step()
                           
                    if features_valid!=None:
                        for X_batch_val, y_batch_val,w_batch_val in loader_val:
                            with torch.no_grad():
                            # validation data
                                if device=="cuda":
                                    Y_hat = model(X_batch_val.cuda())
                                    Y_hat=Y_hat.cpu()
                                    
                                else:
                                    Y_hat = model(X_batch_val)
                                if self.metric=="WMSE":
                                    loss = criterion(Y_hat,y_batch_val,w_batch_val)
                                else:
                                    loss = criterion(Y_hat, y_batch_val)
                                valid_loss+=loss.item()

                    if features_valid!=None:
                        if es(model, valid_loss/len(loader_val)):
                            done = True               
                    else:
                        if es(model, train_loss/len(loader_train)):
                            done =True 

                    history["train_loss"].append(train_loss/len(loader_train))
                    if features_valid!=None:
                        history["val_loss"].append(valid_loss/len(loader_val))

            return history,model
    def dnn_cross_validation(self):
        
        features_train, features_valid, labels_train, labels_valid, weights_train, weights_valid = train_test_split(self.data.features.toarray(),self.data.labels,self.weights, random_state=self.seed)
            


        features_train=torch.tensor(features_train,dtype=torch.float32)
        labels_train=torch.tensor(labels_train,dtype=torch.float32).reshape(-1, 1)
        features_valid=torch.tensor(features_valid,dtype=torch.float32)
        labels_valid=torch.tensor(labels_valid,dtype=torch.float32).reshape(-1, 1)
        weights_train=torch.tensor(weights_train,dtype=torch.float32)
        weights_train=torch.tensor(weights_train,dtype=torch.float32).reshape(-1, 1)
        

                        
        hyperparameters = self.h_parameters

        # Build grid for all parameters
        parameter_grid = itertools.product(*hyperparameters.values())

        grid_search_results = dict()

        for i, grid_comb in enumerate(parameter_grid):
            # Start Grid-Search
            nn_layers, dropout_rate, activation, learning_rate, train_epochs = grid_comb
            
            history,_=self.dnn_train(features_train,
                                     labels_train,
                                     features_valid,
                                     labels_valid,
                                     nn_layers, 
                                     dropout_rate, 
                                     activation, 
                                     learning_rate, 
                                     train_epochs,
                                      weights_train,weights_valid)

            grid_search_results[grid_comb]=history
        
        return grid_search_results       
            
    def dnn_select_best_parameters(self):

        """
        Grid Search selection of best parameters

        """
        grid_search_results = []

        for param_comb, fitted_model in self.cv_results.items():
            # Model Training history
            history_data = pd.DataFrame(fitted_model)
            
            # importing package 
            
            # Save stopping epoch
            history_data = history_data.reset_index().rename(columns={"index": "on_epoch"})
            history_data["on_epoch"] = history_data["on_epoch"].apply(lambda x: x + 1)
 
            # Select loss with minimum validation loss
            best_per_model = history_data.loc[history_data["val_loss"].idxmin(skipna=True)].rename(param_comb)
            
            grid_search_results.append(best_per_model)

        # Concatenate all models training results
        grid_search_results = pd.concat(grid_search_results, axis=1).T

        # select optimal hyperparameter settings
        optimal_stats = grid_search_results.loc[grid_search_results["val_loss"].idxmin(skipna=True)]
        opt_setting = {k: v for k, v in zip(self.h_parameters.keys(), optimal_stats.name)}
       
        opt_setting.update(optimal_stats.to_dict())

        return opt_setting

    def final_model(self):
        features=torch.tensor(self.data.features.toarray(),dtype=torch.float32)
        labels=torch.tensor(self.data.labels,dtype=torch.float32).reshape(-1, 1)
        weights=torch.tensor(self.weights,dtype=torch.float32).reshape(-1, 1)

        _,best_model= self.dnn_train(
                             features,
                             labels,
                             None,
                             None,
                             self.best_params["module__layers"],
                             self.best_params["module__dropout_rate"],
                             self.best_params["module__activation"],
                             self.best_params["optimizer__lr"],
                             int(self.best_params["on_epoch"]),
                             weights,
                             None
                             )

        return best_model  

from ml_utils import SearchModel, MultivariateGaussian
from ml_utils import get_data_below_percentile
import oracles
import mbo
class Oracle_af:
    def __init__(self, data, cv_fold=10,seed=2002):
        self.data = data
        
        percentile = 80
        
       
        #Xtrain_nxm, gttrain_n, _ = get_data_below_percentile(data.features, data.labels, percentile, seed=seed)
        #gt_sigma = 1.0
        #n_train = gttrain_n.size
        
        init_searchmodel = MultivariateGaussian(data.features.shape[1])
        init_searchmodel.fit(data.features)
        
        _, input_dim =data.features.shape
        n_neuralnets = 3 #3
        hidden_units = (100, 100, 100, 100, 10)
        oracle = oracles.DeepEnsemble(input_dim=input_dim, n_nn=n_neuralnets, hidden_units=hidden_units)
        oracle.fit(data.features, data.labels)

        mbo_hp=0.9
        
        n_iter = 20 #20 
        iw_alpha = 0.2
        designalgaf = mbo.ConditioningByAdaptiveSampling(mbo_hp)
        self.weights= designalgaf.run(data.features, data.labels, None, oracle, init_searchmodel, autofocus=True,
                                  iw_alpha=iw_alpha, n_iter=n_iter)
        
    


class Model_Evaluation:
    def __init__(self, model, data, data_idx, metric, smiles=None, data_label=None, data_transformer=None, model_id=None, reg_class="regression",device="cuda"):
        self.reg_class = reg_class
        self.model_id = model_id
        self.metric=metric
        self.data_transformer = data_transformer
        self.smiles=smiles
        self.model = model
        self.data = data
        self.data_idx=data_idx
        self.data_labels = data_label
        self.device=device
        self.labels, self.y_pred, self.predictions = self.model_predict(data)
        self.pred_performance = self.prediction_performance(data)

    def model_predict(self, data):
        
        if self.model_id == 'GCN': #and self.metric!="SERA"
            y_prediction = self.data_transformer.untransform(self.model.predict(data).flatten())
            labels = self.data_labels

    
        #elif self.model.ml_algorithm == 'DNN' and self.metric!="SERA":
        #    y_prediction = self.model.model.predict(data.features.toarray(), verbose=0).flatten()
        #    labels = data.labels
        
        elif self.model.ml_algorithm=="DNN": #and self.metric=="SERA"
            with torch.no_grad():
                features=torch.tensor(self.data.features.toarray(),dtype=torch.float32)
                if self.device=="cuda":
                    y_prediction = self.model.model(features.cuda()).flatten().cpu()
                else:
                    y_prediction = self.model.model(features).flatten()
                y_prediction=[y.item() for y in y_prediction]
                labels=data.labels
        else:
            
            y_prediction = self.model.model.predict(data.features)
            
            labels = data.labels
            
            
            #probs=self.model.model.predict_proba(data.features)
            #print(probs)

        predictions = pd.DataFrame(list(zip(labels, y_prediction)), columns=["true", "predicted"])

        if self.model_id == 'GCN':
        
            predictions['Target ID'] = data.ids
            predictions['algorithm'] = self.model_id
            predictions['MetricOpt']=self.metric
            predictions['Molecules']=self.smiles
            predictions['Index']=self.data_idx
        else:
            predictions['Target ID'] = data.target[0]
            predictions['algorithm'] = self.model.ml_algorithm
            predictions['MetricOpt']=self.metric
            predictions['Molecules']=data.smiles
            predictions['Index']=self.data_idx


        return labels, y_prediction, predictions

    def prediction_performance(self, data, y_score=None, nantozero=False) -> pd.DataFrame:

        if self.reg_class == 'classification':
            fill = 0 if nantozero else np.nan
            if sum(self.y_pred) == 0:
                mcc = fill
                precision = fill
            else:
                mcc = metrics.matthews_corrcoef(data.labels, self.y_pred)
                precision = metrics.precision_score(data.labels, self.y_pred)

            result_list = [{"MCC": mcc,
                            "F1": metrics.f1_score(data.labels, self.y_pred),
                            "BA": metrics.balanced_accuracy_score(data.labels, self.y_pred),
                            "Precision": precision,
                            "Recall": metrics.recall_score(data.labels, self.y_pred),
                            "Average Precision": metrics.average_precision_score(data.labels, self.y_pred),
                            "data_set_size": data.labels.shape[0],
                            "true_pos": len([x for x in data.labels if x == 1]),
                            "true_neg": len([x for x in data.labels if x == 0]),
                            "predicted_pos": len([x for x in self.y_pred if x == 1]),
                            "predicted_neg": len([x for x in self.y_pred if x == 0])},
                           ]

            if y_score is not None:
                result_list.append({"AUC": metrics.roc_auc_score(data.labels, self.y_pred)})
            else:
                result_list.append({"AUC": np.nan})

            results = pd.DataFrame(result_list)
            results = results[
                ["Target ID", "Algorithm", "MCC", "F1", "BA", "Precision", "Recall", "Average Precision"]]
            results["Target ID"] = results["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
            results.set_index(["Target ID", "Algorithm"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"],
                                                          ["MCC", "F1", "BA", "Precision", "Recall",
                                                           "Average Precision"]], names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results

        elif self.reg_class == "regression":

            labels = self.labels
            pred = self.y_pred

            fill = 0 if nantozero else np.nan
            if len(pred) == 0:
                mae = fill
                mse = fill
                rmse = fill
                r2 = fill
                sera=fill

            else:
                mae = mean_absolute_error(labels, pred)
                mse = metrics.mean_squared_error(labels, pred)
                rmse = metrics.mean_squared_error(labels, pred, squared=False)
                r2 = metrics.r2_score(labels, pred)
                
               

            if self.model_id == 'GCN':
                target =data.ids[0] # data[0].id
                model_name = self.model_id
            else:
                target = data.target[0]
                model_name = self.model.ml_algorithm

            result_dict = {"MAE": mae,
                            "MSE": mse,
                            "RMSE": rmse,
                            "R2": r2,
                            "data_set_size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name,
                            "Selection Metric": self.metric}
                           
           
           
          
     
            ph={"method":"range","npts": 3,"control_pts":[0,0,0,8,0.1,0,9,1,0]}
            phi_labels=SERA_phi_control(labels,ph)
            sera=sera_pt(torch.Tensor(labels),torch.Tensor(pred),phi_labels)/len(labels)
            result_dict["SERA"]=float(np.array(sera))

            result_list=[result_dict]

            # Prepare result dataset
            results = pd.DataFrame(result_list)
            results = results[["Target ID", "Algorithm", "Selection Metric","MAE", "MSE", "RMSE", "R2","SERA"]]
            results["Target ID"] = results["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
            results.set_index(["Target ID", "Algorithm", "Selection Metric"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2","SERA"]],
                                                         names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results


