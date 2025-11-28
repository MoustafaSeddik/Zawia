# %% Importing the necessary libraries

#%% import src libraries
from src.data_preprocessing.preprocess import load_datasets, prepare_data
from src.training.train import model_train
from src.config.hyperparameter_tuning import run_optuna, report_study
from src.settings.settings import *
from src.utils.genPlots import *
from src.inference.Evaluate import *
from src.utils.helpers import (ScatterSurfacedata,
                               setget_model_path, study_results)
from src.inference.Predict_plots import plot_residual, plot_residual_hist, normalQQ_plot
from src.data_preprocessing.EDA_plots import plot_density_distributions
#%% checking setting
print(f"{Fore.RED}mode: {Style.RESET_ALL}{mode}")
print(f"{Fore.RED}type of dataset: {Style.RESET_ALL}{features_set}")
print(f"{Fore.RED}Iteration number:: {Style.RESET_ALL}{iteration}")
print(f"{Fore.RED}Show the study visualizations: {Style.RESET_ALL}{study_vis}")
print(f"{Fore.RED}predict: {Style.RESET_ALL}{predict}")
print(f"{Fore.RED}Show the Prediction visualizations: {Style.RESET_ALL}{predict_vis}")

#%% Load and prepare the datasets

df_train, df_val, df_test = load_datasets(train_filename,
                                          target_filename,
                                          train_size=0.7,
                                          val_size=0.2)

(X_train, y_train, X_val, y_val, X_test, y_test,
 testData, fig) = prepare_data(df_train,
                          df_val, df_test,
                          normalization=normalization,
                          visualization="Training")

save_plot(fig, plot_name=f"{features_set}_BoxPlot",
          experiment_name=f"{features_set}",
          base_output_dir=f"training_results")

fig = plot_density_distributions(X_train, X_val, X_test)

#%% Run/Load the hyperparameter Study
study = None
trial_number = None
param, history = None, {}

if mode == "tuning":
    # Create a new study and run the optimization
    study = run_optuna(num_trails=100,
               study_name = study_name,
               database_name = database_name,
               X_train = X_train,
               y_train = y_train,
               X_val = X_val,
               y_val = y_val)

elif mode == "predict":
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(f"{Fore.RED}Study '{study_name}' loaded successfully!{Style.RESET_ALL}")
    report_study(study)
    trial_number = study.trials[study.best_trial.number].number

    study_df = study_results(study)
    study_df_fig = plot_study_df(study_df)
    save_plot(fig=study_df_fig, plot_name= "study_df",
              base_output_dir=f"{study_folder_name}/{study_name}",
              experiment_name=" "
    )
elif mode == "training":
    # Load the study from the database
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(f"{Fore.RED}Study '{study_name}' loaded successfully!{Style.RESET_ALL}")
    print(f"{Fore.RED}Total number of trials in the study:{Style.RESET_ALL} {len(study.trials)}")

    #  Train the model further using the best hyperparameters found by Optuna
    trial_number = study.trials[study.best_trial.number].number
    print(f"{Fore.BLUE}loaded trial number: {Style.RESET_ALL}{trial_number}")
    param = {
        'epochs': 120, # Number of epochs to train the model
        "loss function": "huber",
        'lr': study.trials[trial_number].params['lr']*1,  # Learning rate for the optimizer
        'batch size': study.trials[trial_number].params['batch size']*1,  # Batch size for training
        'weight decay': study.trials[trial_number].params['weight decay'],  # Regularization parameter
        'hidden_layers': study.trials[trial_number].params['hidden_layers'],  # Number of hidden layers in the model
        'hidden_units': study.trials[trial_number].params['hidden_units'],  # Number of hidden units in each layer
        'width_type': study.trials[trial_number].params['width_type'],  # 'constant', 'upward', or 'downward'
        'drop_out': study.trials[trial_number].params['drop_out']  # Dropout rate to prevent overfitting
    }
    (history, model, bestModel) = model_train(X_train, y_train, X_val, y_val,
                                              params=param, trial= None)

    # Save the best model
    if trial_number == study.best_trial.number:
        torch.save(bestModel.state_dict(), best_model_path)
        print(f"{Fore.RED}Model saved to: {Style.RESET_ALL}{best_model_path}")
    else:
        # Save the best model of this trial
        trial_model_path = setget_model_path(trial_number, itr="Adam")
        torch.save(bestModel.state_dict(), trial_model_path)
        print(f"{Fore.RED}Best Model of trial: {Style.RESET_ALL}{trial_number} "
              f"{Fore.RED}saved to: {Style.RESET_ALL}{trial_model_path}")

#%% Visualization of results
if study_vis:
    if mode == "training":
        hist_plot= plot_hist(history, params=param)
        save_plot(fig=hist_plot, plot_name= "Adam_training_history",
                  base_output_dir=f"{trials_dir}/{study.trials[trial_number].number}",
                  experiment_name=" "
                  )
        print()
        print(f"number of parameters:{history.get('num_params')}")
        print(f"Training time: {history.get('training_time')}")
        print(f"tm_2:\n{history.get('tm_2')}")
        print(f"ep_2:\n{history.get('ep_2')}")

    elif mode == "tuning" or mode == "predict":
        visualize_study(study)
        #plot_study_history(study, start=len(study.trials) - 10, end=len(study.trials))
        #plot_param_frequencies(study, bins_for_float=50, max_unique=70, round_float=20)
        #plot_param_frequencies_interactive(study, max_unique=20, round_float=3, bins_for_float=20)
        #history = study.trials[study.best_trial.number].user_attrs.get("history")
        #hist_plot = plot_hist(history, params=study.best_trial.params)
        #save_plot(fig=hist_plot, plot_name= "training_history",
        #          base_output_dir=f"{trials_dir}/{study.trials[study.best_trial.number].number}",
        #          experiment_name=" "
        #         )
        parallelCoordinatePlot(study_df, leading_column = "nMAE",
                               colorbar=True,
                               style="viridis",  # "viridis", "plasma", "inferno", "magma", "cividis",
                               interpolation='linear',
                               cross_val=True)
    elif mode == "analysis":
        fig = Plot_feature_sets(iteration="_4", trials=10)
        save_plot(fig=fig, plot_name="nMAE_vs_duration",
                  base_output_dir=f"optuna_results",
                  experiment_name="iteration 4"
                  )

#%% Predict using the best model
if mode == "predict" or predict == True:
    model_path = setget_model_path(trial_number, itr="Adam")
    loaded_model = load_model(X_train, study, model_path, trial_number)
    results_df   = evaluate_model(loaded_model, y_test ,testData)
    if predict_vis:
        # plot true vs. predicted 2D plot
        true_pred_plot = plot_true_pred(results_df)
        save_plot(fig=true_pred_plot, plot_name="true_pred_plot",
                  base_output_dir=f"{trials_dir}/{study.trials[trial_number].number}",
                  experiment_name=" "
                  )

        residual_plot = plot_residual(results_df)
        save_plot(fig=residual_plot, plot_name="residual_plot",
                  base_output_dir=f"{trials_dir}/{study.trials[trial_number].number}",
                  experiment_name=" "
                  )

        residual_hist = plot_residual_hist(results_df)
        save_plot(fig=residual_hist, plot_name="residual_hist",
                  base_output_dir=f"{trials_dir}/{study.trials[trial_number].number}",
                  experiment_name=" "
                  )

        normalQQ_plot = normalQQ_plot(results_df)
        save_plot(fig=normalQQ_plot, plot_name="normalQ_Q_plot",
                  base_output_dir=f"{trials_dir}/{study.trials[trial_number].number}",
                  experiment_name=" "
                  )
        # plotting 3D plot
        (zenScatter, aziScatter, rScatter,
         zenSurface, aziSurface, rSurface) = ScatterSurfacedata(filename=df_train,
                                                                uFac=100, gridsize=500,
                                                                loaded_model=loaded_model,
                                                                testData=testData)

        plotScatterSurface(zenScatter, aziScatter, rScatter,
                           zenSurface, aziSurface, rSurface, show=True,
                           fn=f"{trials_dir}/{study.trials[trial_number].number}/3D_surface_plot.html"
                           )

#%% Exploring new ideas
if mode == "explore" :
    param = {
        'epochs': 120, # Number of epochs to train the model
        "loss function": "huber",
        'lr': 1e-4,  # Learning rate for the optimizer
        'batch size': 512,  # Batch size for training
        'weight decay': 1e-5,  # Regularization parameter
        'hidden_layers': 5,  # Number of hidden layers in the model
        'hidden_units': 900,  # Number of hidden units in each layer #750
        'width_type':"constant",  # 'constant', 'upward', or 'downward' #downward
        'drop_out':0.2 # Dropout rate to prevent overfitting
    }
    (history, model, bestModel) = model_train(X_train, y_train, X_val, y_val,
                                              params=param, trial= None,
                                              optim = "Adam")
    parent_dir = os.path.dirname(best_model_path)
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(bestModel.state_dict(), best_model_path)
    print(f"{Fore.RED}Model saved to: {Style.RESET_ALL}{best_model_path}")

    hist_plot = plot_hist(history, params=param)
    save_plot(fig=hist_plot, plot_name=f"Adam_training_history",
              base_output_dir=f"{trainings_folder_name}/{features_set}",
              experiment_name=" "
              )
    # Predict using the best model
    loaded_model = load_model(X_train = X_train,
                              study = None,
                              model_path = best_model_path,
                              trial_number = param)
    results_df   = evaluate_model(loaded_model, y_test ,testData)

    true_pred_plot = plot_true_pred(results_df)
    save_plot(fig=true_pred_plot, plot_name="true_pred_plot",
              base_output_dir=f"{trainings_folder_name}/{features_set}",
              experiment_name=" "
              )
