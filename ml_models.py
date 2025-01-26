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

model_list = ['SVR','kNN','XGBoost','DNN']
cv_folds=5

selection_metric =["MAE","MSE","SERA"]

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

# Generate Molecular Fingerprints
morgan_radius2 = FoldedMorganFingerprint(radius=2)
morgan_radius2.fit_smiles(regression_db.nonstereo_aromatic_smiles.tolist())



for target in regression_tids:
    for model in model_list:
        print(model)
        performance_train_df = pd.DataFrame()
        predictions_train_df = pd.DataFrame()
        performance_test_df = pd.DataFrame()
        predictions_test_df = pd.DataFrame()
        parameter_resume = []
        result_path = create_directory(f'./regression_results/{target}/{model}')
        # Final Dataframes

        print(f'Training on {target}')

        # Select Target Database
        regression_db_tid = regression_db.loc[regression_db.chembl_tid == target]

        dev=stats.stdev(regression_db_tid.pPot.values)

        # Constructing ChEMBL Dataset
        fp_matrix = morgan_radius2.transform_smiles(regression_db_tid.nonstereo_aromatic_smiles.tolist())
        
        # Constructing Dataset
        
        dataset = Dataset(fp_matrix, np.array(regression_db_tid.pPot.values))
        dataset.add_instance("target", regression_db_tid.chembl_tid.values)
        dataset.add_instance("smiles", regression_db_tid.nonstereo_aromatic_smiles.values)
            

        
        # Split dataset into TR and TE
        data_splitter = KFold(n_splits=cv_folds, random_state=20021997, shuffle=True)
        for trial, (train_idx, test_idx) in enumerate(data_splitter.split(dataset.features, dataset.target)):
            print(trial)
            
            #Defining Training and Test sets
            training_set = dataset[train_idx]
            test_set = dataset[test_idx]
            
            # set seed
            set_global_determinism(seed=trial)
 
            for sel_metric in selection_metric:
                print(f'Training {model}-{sel_metric}')
                # Save ML models
                #model_fpath = create_directory(f"./regression_results/trained_models/{model}_{opt}/", verbose=False)
                
               
                ph={"method":"range","npts": 3,"control_pts":[0,0,0,8,0.1,0,9,1,0]}

                #if model == 'DNN' and sel_metric!="SERA":
                #    ml_model = DNN(training_set, model, training_set.features.shape[1], seed=trial, metric=sel_metric)
                #    #model_fpath += ".h5"
                #    #ml_model.model.save(model_fpath)
                if model == 'DNN': #and sel_metric=="SERA"
                    ml_model = DNN(training_set, model, training_set.features.shape[1], seed=trial,  ph=ph, metric=sel_metric)
                    #model_fpath += "model.pth"
                    #torch.save(ml_model.model,model_fpath)
                
                else:
                    ml_model = MLModel(training_set, model, metric=sel_metric,ph=ph)
                    #model_fpath += ".sav"
                    #pickle.dump(ml_model, open(model_fpath, 'wb'))

                #Best model parameters dictionary
                opt_parameters_dict = {'model': model,
                                    'trial': trial,
                                    'Target ID': target,
                                    'Selection Metric': sel_metric,
                                    
                                    }
                for param, value in ml_model.best_params.items():
                    opt_parameters_dict[param] = value
                parameter_resume.append(opt_parameters_dict)

                # TRAIN
                #Model Evaluation
                model_eval_train = Model_Evaluation(ml_model, training_set, train_idx, sel_metric)

                #Performance df
                performance_train = model_eval_train.pred_performance
                performance_train["trial"] = trial
              
                performance_train_df = pd.concat([performance_train_df, performance_train])

                # TEST
                #Model Evaluation
                model_eval_test = Model_Evaluation(ml_model, test_set, test_idx, sel_metric)

                #Performance df
                performance_test = model_eval_test.pred_performance
                performance_test["trial"] = trial
                performance_test_df = pd.concat([performance_test_df, performance_test])

                # Prediction df
                predictions_test = model_eval_test.predictions
                predictions_test["trial"] = trial
                predictions_test_df = pd.concat([predictions_test_df, predictions_test])


        parameter_df = pd.DataFrame(parameter_resume)

        performance_train_df.to_csv(os.path.join(result_path, f'performance_train.csv'))
        performance_test_df.to_csv(os.path.join(result_path, f'performance_test.csv'))
        parameter_df.to_csv(os.path.join(result_path, f'model_best_parameters.csv'))
        predictions_test_df.to_csv(os.path.join(result_path, f'predictions_test.csv'))

# %%
