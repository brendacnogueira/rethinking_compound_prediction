# %%
"""
# Load Libraries
"""

# %%
#Utils
from ml_utils import *
from machine_learning_models import *
from fingerprints import *
import random

#Sklearn
from sklearn.model_selection import KFold 
import statistics as stats

import tensorflow as tf
#deepchem
import deepchem as dc
from deepchem.models.torch_models import GCNModel
from deepchem.models.losses import  L1Loss,L2Loss, Loss, _make_pytorch_shapes_consistent, _make_tf_shapes_consistent, _ensure_float
import pickle
from descriptors import CalcMolDescriptors
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#%load_ext autoreload
#%autoreload 2

# %%
"""
# Models Parameters
### Select the desired parameters to be used by regression models
model_list: ML/DL models for regression (kNN: k-neirest neighbor, SVR: Support Vector Regression, RFR: Random Forest Regression, DNN: Deep Neural Network, MR: Median regression)
cv_fold: Number do data splits (trials) to be performed
selection_metric: metric to loss function and hyperparameter optimization (MAE: 'neg_mean_absolute_error', MSE: ‘neg_mean_squared_error’)



"""

# %%

model_list = ['GCN','DNN','XGBoost','SVR','kNN'] 
cv_folds=5

selection_metric =["AFO"]


# %%
"""

# Load Data
### Load compound database to be used for the regression models

db_path: dataset full path
"""

# %%
# Database path
db_path = './dataset/'
# Load actives dB
regression_db = pd.read_csv(os.path.join(db_path, f'chembl_30_IC50_10_tids_1000_CPDs.csv'))
# Regression Compound Targets
regression_tids = regression_db.chembl_tid.unique()[:10]

# %%
"""
# Models
"""

# %%

morgan_radius2 = FoldedMorganFingerprint(radius=2)
morgan_radius2.fit_smiles(regression_db.nonstereo_aromatic_smiles.tolist())

params_dict = {
     "nb_epoch":[100, 200],
     "learning_rate":[0.01, 0.001],
     "n_tasks":[1],
     "graph_conv_layers":[[64, 64], [256, 256], [512, 512], [1024, 1024]],
     "dense_layer_size":[64, 256, 512, 1024],
     "dropout":[0.0],
     "mode":["regression"],
     "number_atom_features":[30]}

for target in regression_tids:
    print(f'Training on {target}')
    result_path = create_directory(f'./regression_results/{target}/Oracle')
    weight_resume=pd.DataFrame()
    performance_train_df = pd.DataFrame()
    predictions_train_df = pd.DataFrame()
    performance_test_df = pd.DataFrame()
    predictions_test_df = pd.DataFrame()
    parameter_resume = []
     # Select Target Database
    regression_db_tid = regression_db.loc[regression_db.chembl_tid == target]
    dev=stats.stdev(regression_db_tid.pPot.values)

    # Data for oracle
    mol_objects = [Chem.MolFromSmiles(smile) for smile in regression_db_tid.nonstereo_aromatic_smiles.values]
        
    features =pd.DataFrame([CalcMolDescriptors(x) for x in mol_objects])
    features=features.drop(columns=["MaxAbsEStateIndex","MinAbsEStateIndex","MaxAbsPartialCharge","MinAbsPartialCharge","NumRadicalElectrons","setDescriptorVersion","defaultdict"]) #'SlogP_VSA9','SlogP_VSA7'
    #features=features.drop(features.columns[8], axis=1) #features.columns[[8,66]]
    
    columns=features.columns[:13].append(features.columns[15:40])
    features=features[columns]
    features=(features-features.min())/(features.max()-features.min())
    
    features=features.to_numpy()
    dataset_oracle = Dataset(features, np.array(regression_db_tid.pPot.values))

    

    for model in model_list: 

        print(model)
        sel_metric="AFO"

        model_fpath = create_directory(f"./trained_models/{model}_{sel_metric}/", verbose=False)
        # Final Dataframes

        # Constructing ChEMBL Dataset
        
        # Constructing Dataset
        
        if model=="GCN":
            featurizer= dc.feat.MolGraphConvFeaturizer()
            mol = [Chem.MolFromSmiles(smi) for smi in regression_db_tid.nonstereo_aromatic_smiles.tolist()]
            mol_graphs = featurizer.featurize(mol)
            
            potency = regression_db_tid.pPot.values.tolist()
            dataset = dc.data.NumpyDataset(X=mol_graphs, y=np.array(potency), ids=np.array(regression_db_tid.chembl_tid.values))
            split_dat=dataset.X
        else:  
            fp_matrix = morgan_radius2.transform_smiles(regression_db_tid.nonstereo_aromatic_smiles.tolist())

            dataset = Dataset(fp_matrix, np.array(regression_db_tid.pPot.values))
            dataset.add_instance("target", regression_db_tid.chembl_tid.values)
            dataset.add_instance("smiles", regression_db_tid.nonstereo_aromatic_smiles.values)
            split_dat=dataset.features #, dataset.target
       
        
        # Split dataset into TR and TE
        data_splitter = KFold(n_splits=cv_folds, random_state=20021997, shuffle=True)
        
        for trial, (train_idx, test_idx) in enumerate(data_splitter.split(split_dat)):
            print(trial)
            print(f'Training {model}-{"AFO"}')
        
            training_set_oracle= dataset_oracle[train_idx]
            
            # set seed
            set_global_determinism(seed=trial)
            # Save ML models
            model_fpath = create_directory(f"./trained_models/{target}/{model}_{sel_metric}/", verbose=False)
            
            ph={"method":"range","npts": 3,"control_pts":[0,0,0,8,0.1,0,9,1,0]}

            weights=Oracle_af(training_set_oracle).weights
            
        
            weights_resume_dict = pd.DataFrame({
                                'Weights': weights,
                                'Labels':training_set_oracle.labels
                                })
            
            weights_resume_dict["Target ID"]=target
            weights_resume_dict["trial"]=trial
            weight_resume = pd.concat([weight_resume, weights_resume_dict])

           # continue

            


            opt_parameters_dict = {'model': model,
                                'trial': trial,
                                'Target ID': target,
                                'Selection Metric': sel_metric,
                              
                                }
            
            if model=="GCN":
                #training_set_u = dataset.select(train_idx)
                #test_set_u = dataset.select(test_idx)
               
             
                training_set_u =dc.data.NumpyDataset(X=mol_graphs[train_idx], y=np.array(regression_db_tid.pPot.values[train_idx].tolist()),w=np.array(weights), ids=np.array(regression_db_tid.chembl_tid.values[train_idx]))
                test_set_u =dc.data.NumpyDataset(X=mol_graphs[test_idx], y=np.array(regression_db_tid.pPot.values[test_idx].tolist()), ids=np.array(regression_db_tid.chembl_tid.values[test_idx]))


                smiles_train=regression_db_tid.iloc[train_idx].nonstereo_aromatic_smiles.values
                smiles_test=regression_db_tid.iloc[test_idx].nonstereo_aromatic_smiles.values
                
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
                
                metric_dict={"MSE":dc.metrics.mean_squared_error} 
                loss_dict={"MSE":L2Loss()} 
            
                metric_name= metric_dict["MSE"]
                metric = dc.metrics.Metric(metric_name,mode="regression")
                
                
                params_dict["loss_func"]=[loss_dict["MSE"]]
                
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
                
                ml_model = final_gcn(training_set, best_params)
               # gcn_path=model_fpath+"graphconv_model.pth"

                model_eval_train = Model_Evaluation(ml_model, training_set, train_idx, sel_metric, smiles_train,
                                                        training_set_u.y, transformers[0], model_id=model)
                
                model_eval_test = Model_Evaluation(ml_model, test_set,test_idx, sel_metric, smiles_test,
                                                        test_set_u.y, transformers[0], model_id=model)
               
                del ml_model,best_model, best_params, all_results
            else:
                   #Defining Training and Test sets
                training_set = dataset[train_idx]
                test_set = dataset[test_idx]

                if  model == "DNN":
                    ml_model = DNN(training_set, model, training_set.features.shape[1], seed=trial,  ph=ph, metric="WMSE", weights=weights)
                    #model_fpath += "model.pth"
                    #torch.save(ml_model.model,model_fpath)

                else:
                    ml_model = MLModel(training_set, model, metric="MSE",ph=ph,weights=weights)
                    #model_fpath += ".sav"
                    #pickle.dump(ml_model, open(model_fpath, 'wb'))
                #Best model parameters dictionary
           
                for param, value in ml_model.best_params.items():
                    opt_parameters_dict[param] = value
                parameter_resume.append(opt_parameters_dict)

                model_eval_train = Model_Evaluation(ml_model, training_set, train_idx, sel_metric)
                model_eval_test = Model_Evaluation(ml_model, test_set, test_idx, sel_metric)

            # TRAIN
            #Performance df
            performance_train = model_eval_train.pred_performance
            performance_train["trial"] = trial

            performance_train_df = pd.concat([performance_train_df, performance_train])

            # TEST
            #Performance df
            performance_test = model_eval_test.pred_performance
            performance_test["trial"] = trial
          
            performance_test_df = pd.concat([performance_test_df, performance_test])

            # Prediction df
            predictions_test = model_eval_test.predictions
            predictions_test["trial"] = trial
            predictions_test_df = pd.concat([predictions_test_df, predictions_test])

    
   # weight_resume.to_csv(os.path.join(result_path, f'weights.csv'))
            
    parameter_df = pd.DataFrame(parameter_resume)
    performance_train_df.to_csv(os.path.join(result_path, f'performance_train.csv'))
    performance_test_df.to_csv(os.path.join(result_path, f'performance_test.csv'))
    parameter_df.to_csv(os.path.join(result_path, f'model_best_parameters.csv'))
    predictions_test_df.to_csv(os.path.join(result_path, f'predictions_test.csv'))

# %%
