# %%
#Utils
from ml_utils import *
from machine_learning_models import *
from fingerprints import *
from sklearn.model_selection import KFold
import random
import tensorflow as tf
#deepchem
import deepchem as dc
from deepchem.models.torch_models import GCNModel
from deepchem.models.losses import  L1Loss,L2Loss, Loss, _make_pytorch_shapes_consistent, _make_tf_shapes_consistent, _ensure_float
#from IPython.core.display_functions import display
import warnings
import sys
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
#%load_ext autoreload
#s%autoreload 2
import random

#Sklearn
import statistics as stats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#%load_ext autoreload
#%autoreload 2

# %%


class SeraLoss(Loss):
    def __init__(self, ph,device="cuda"):
        super().__init__()
        self.ph=ph
        self.device=device

    def _create_pytorch_loss(self):
        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            criterion = SeraCriterion(self.ph,device=self.device)
            return criterion(output,labels)
            

        return loss
# %%
"""
# Models Parameters
### Select the desired parameters to be used by regression models
model_list: ML/DL models for regression (kNN: k-neirest neighbor, SVR: Support Vector Regression, RFR: Random Forest Regression, DNN: Deep Neural Network, MR: Median regression)
cv_fold: Number do data splits (trials) to be performed
selection_metric: metric to loss function and hyperparameter optimization (MAE: 'neg_mean_absolute_error', MSE: ‘neg_mean_squared_error’,SERA: Squared Error-Relevance Area)

"""

# %%
model_list = ['GCN']
cv_folds=5
selection_metric = ["MSE","MAE","SERA"] 



#%%



params_dict = {
     "nb_epoch":[100, 200],
     "learning_rate":[0.01, 0.001],
     "n_tasks":[1],
     "graph_conv_layers":[[64, 64], [256, 256], [512, 512], [1024, 1024]],
     "dense_layer_size":[64, 256, 512, 1024],
     "dropout":[0.0],
     "mode":["regression"],
     "number_atom_features":[30]}



# %%
"""
# Load Data
### Load compound database to be used for the regression models

db_path: dataset full path

"""

# %%
# Load actives dB
db_path = './dataset/'
# Load actives dB
regression_db = pd.read_csv(os.path.join(db_path, f'chembl_30_IC50_10_tids_1000_CPDs.csv'))
# Target Classes
regression_tids = regression_db.chembl_tid.unique()[:]

# %%
"""
# GCN Models
### Folowing code generates potency prediction based on GCN models
"""
# %%
#Create saving path
create_directory('./regression_results/')


for target in regression_tids:
    for model in model_list:
        performance_train_df = pd.DataFrame()
        performance_test_df = pd.DataFrame()
        predictions_test_df = pd.DataFrame()
        predictions_train_df = pd.DataFrame()
        parameter_resume = []
        result_path = create_directory(f'./regression_results/{target}/{model}')
        print(f'Training on {target}-{model}')

        # Select Target Database
        regression_db_tid = regression_db.loc[regression_db.chembl_tid == target]


        for sel_metric in selection_metric:
            
                ph={"method":"range","npts": 3,"control_pts":[0,0,0,8,0.1,0,9,1,0]}
                dev=stats.stdev(regression_db_tid.pPot.values)

                # Constructing ChEMBL Dataset
                #dataset=MoleculeDataset(root=f"gcn_data/",data=regression_db_tid,filename="")
                
                # Split dataset into TR and TE
                featurizer= dc.feat.MolGraphConvFeaturizer()
    

                mol = [Chem.MolFromSmiles(smi) for smi in regression_db_tid.nonstereo_aromatic_smiles.tolist()]
                mol_graphs = featurizer.featurize(mol)
               
                potency = regression_db_tid.pPot.values.tolist()
                dataset = dc.data.NumpyDataset(X=mol_graphs, y=np.array(potency), ids=np.array(regression_db_tid.chembl_tid.values))


                # Split dataset into TR and TE
                data_splitter = KFold(n_splits=cv_folds, random_state=20021997, shuffle=True)

                for trial, (train_idx, test_idx) in enumerate(data_splitter.split(dataset.X)):
                
                    #Defining Training and Test sets
                    training_set_u = dataset.select(train_idx)
                    test_set_u = dataset.select(test_idx)
                    smiles_train=regression_db_tid.iloc[train_idx].nonstereo_aromatic_smiles.values
                    smiles_test=regression_db_tid.iloc[test_idx].nonstereo_aromatic_smiles.values

                    # Initialize transformers
                    transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=training_set_u, move_mean=True)]

                    #Transform data
                    for transformer in transformers:
                        training_set = transformer.transform(training_set_u)
                    for transformer in transformers:
                        test_set = transformer.transform(test_set_u)

                    # Split dataset into TR and internal Validation
                    splitter = dc.splits.RandomSplitter()
                    train_set, valid_set = splitter.train_test_split(training_set, seed=trial)

        
                    #Define random seed
                    set_global_determinism(seed=trial)
                    #set_seeds(trial)

                    #Initialize GridSearch optimizer
                    optimizer = dc.hyper.GridHyperparamOpt(dc.models.torch_models.GCNModel)
                    def sera_score(y_true,y_pred,device="cpu"):
                        phi_labels=SERA_phi_control(y_true,ph)
                        sera=sera_pt(torch.Tensor(y_true),torch.Tensor(y_pred),phi_labels,device=device)
                        return float(np.array(sera))

                    metric_dict={"MAE":dc.metrics.mae_score,"MSE":dc.metrics.mean_squared_error,"SERA":sera_score} 
                    loss_dict={"MAE":L1Loss(),"MSE":L2Loss(),"SERA":SeraLoss(ph)} 
                
                    metric_name= metric_dict[sel_metric]
                    metric = dc.metrics.Metric(metric_name,mode="regression")
                    
                    
                    params_dict["loss_func"]=[loss_dict[sel_metric]]
                    
                    # Best GCN model, parameters and final results
                    best_model, best_params, all_results = optimizer.hyperparam_search(params_dict=params_dict,
                                                                                        train_dataset=train_set,
                                                                                        valid_dataset=valid_set,
                                                                                        metric=metric,
                                                                                        use_max=False,
                                                                                        output_transformers=transformers,
                                                                                        logdir='/afs/crc.nd.edu/user/b/bcruznog/ML_Potency/logfiles/gcn/'
                                                                                        )

                    # Define final GCN model

                    def final_gcn(data, best_params):

                        gcn = GCNModel(loss_func=best_params["loss_func"],
                                            n_tasks=best_params["n_tasks"],
                                            graph_conv_layers=best_params["graph_conv_layers"],
                                            dropout=best_params["dropout"],
                                            mode=best_params["mode"],
                                            predictor_hidden_feats=best_params["dense_layer_size"],
                                            learning_rate=best_params["learning_rate"]
                                        )

                        gcn.fit(data, nb_epoch=best_params["nb_epoch"],max_checkpoints_to_keep=1, checkpoint_interval=20000)

                        return gcn
        
                        #Best GCN model parameters
                    opt_parameters_dict = {'model': model,
                                            'trial': trial,
                                            'Target ID': target,
                                            'Selection Metric': sel_metric,
                                           
                                            }

                    if isinstance(best_params, tuple):
                        best_params = {
                            "nb_epoch":best_params[0],
                            "learning_rate":best_params[1],
                            "n_tasks":best_params[2],
                            "graph_conv_layers":best_params[3],
                            "dense_layer_size":best_params[4],
                            "dropout":best_params[5],
                            "mode":best_params[6],
                            "number_atom_features":best_params[7],
                            }

                    for param, value in best_params.items():
                        opt_parameters_dict[param] = value
                    parameter_resume.append(opt_parameters_dict)

                    # Generate final Model
                    ml_model = final_gcn(training_set, best_params)

                    # evaluate the model
                    train_score = ml_model.evaluate(training_set, [metric], transformers)
                    test_score = ml_model.evaluate(test_set, [metric], transformers)
                    print(test_score)
                        #TRAIN
                    #Model Evaluation
                    model_eval_train = Model_Evaluation(ml_model, training_set, train_idx, sel_metric, smiles_train,
                                                        training_set_u.y, transformers[0], model_id=model)

                    #Performance df
                    performance_train = model_eval_train.pred_performance
                    performance_train["trial"] = trial
                    performance_train["Y"] = Y
                    performance_train_df = pd.concat([performance_train_df, performance_train])

                    # Prediction df
                    predictions_train = model_eval_train.predictions
                    predictions_train["trial"] = trial
                    predictions_train_df = pd.concat([predictions_train_df, predictions_train])

                    #Model Evaluation
                    model_eval_test = Model_Evaluation(ml_model, test_set,test_idx, sel_metric, smiles_test,
                                                        test_set_u.y, transformers[0], model_id=model)

                    #Performance df
                    performance_test = model_eval_test.pred_performance
                    performance_test["trial"] = trial
                    performance_test_df = pd.concat([performance_test_df, performance_test])

                    # Prediction df
                    predictions_test = model_eval_test.predictions
                    predictions_test["trial"] = trial
                    predictions_test_df = pd.concat([predictions_test_df, predictions_test])

                    del ml_model,best_model, best_params, all_results
                

    
        parameter_df = pd.DataFrame(parameter_resume)

        # Save results

        performance_train_df.to_csv(os.path.join(result_path, f'performance_train.csv'))
        performance_test_df.to_csv(os.path.join(result_path, f'performance_test.csv'))
        parameter_df.to_csv(os.path.join(result_path, f'model_best_parameters.csv'))
        predictions_test_df.to_csv(os.path.join(result_path, f'predictions_test.csv'))
        predictions_train_df.to_csv(os.path.join(result_path, f'predictions_train.csv'))
