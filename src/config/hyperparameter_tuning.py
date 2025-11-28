import os
import optuna
import pandas as pd
from src.training.train import model_train
from src.settings.settings import *
from src.utils.genPlots import plot_hist, save_plot
from src.utils.helpers import setget_model_path

# %% Tuning the model hyperparameters
def objective_factory(X_train, y_train, X_val, y_val):
    def objective(trial):
        # Define the hyperparameters to tune
        param = {
            'epochs': 100,  # Number of epochs to train the model
            #"loss function": trial.suggest_categorical("loss function", ["smooth_l1", "huber", "MSE", "MAE"]),
            "loss function": "smooth_l1",
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),  # Learning rate for the optimizer
            'batch size': trial.suggest_categorical('batch size', [64, 128, 256, 512]),  # Batch size for training
            'weight decay': trial.suggest_float('weight decay', 1e-6, 1e-3, log=True),  # Regularization parameter
            'hidden_layers': trial.suggest_int('hidden_layers', 1, 10),  # Number of hidden layers
            'hidden_units': trial.suggest_int('hidden_units', 32, 1024),  # Number of units in each layer
            'width_type': trial.suggest_categorical('width_type', ['constant', 'downward', 'upward']),  # Width type
            'drop_out': trial.suggest_float('drop_out', 0.0, 0.5)  # Dropout rate
        }

        # Train the model
        history, _, bestModel = model_train(
            X_train, y_train,
            X_val, y_val,
            params=param, trial=trial
        )

        # Save the best model of this trial
        trial_model_path = setget_model_path(trial.number, itr="Adam")
        torch.save(bestModel.state_dict(), trial_model_path)
        print(f"{Fore.RED}Best Model of trial: {Style.RESET_ALL}{trial.number} "
              f"{Fore.RED}saved to: {Style.RESET_ALL}{trial_model_path}")

        # save the history plot
        hist_plot = plot_hist(history, params=trial.params)
        save_plot(fig=hist_plot, plot_name= "training_history",
                  base_output_dir= f"{trials_dir}/{trial.number}",
                  experiment_name=" "
                  )
        # Return the last validation loss trained in this trial
        return history.get('val_loss_hist')[-1]
    return objective

def run_optuna(num_trails, study_name, database_name, X_train, y_train, X_val, y_val):
    # Use TPE sampler for better exploration
    TPE_sampler = optuna.samplers.TPESampler(n_startup_trials=10,
                                             multivariate=True,
                                             seed=42)

    # Use Hyperband pruner for efficient pruning
    pruner = optuna.pruners.HyperbandPruner(min_resource=25,
                                            max_resource='auto',
                                            reduction_factor=2)

    # Create a study object with the TPE sampler and Hyperband pruner
    results_dir = study_folder_name
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    db_path = os.path.join(results_dir, database_name)
    storage_url = f"sqlite:///{db_path}"

    # Define the study name. This allows you to store multiple studies in one database file.
    print(f"Creating/loading study '{study_name}' from storage: {storage_url}")
    study = optuna.create_study(direction='minimize',
                                sampler=TPE_sampler,
                                pruner=pruner,
                                study_name=study_name,
                                storage=storage_url,
                                load_if_exists=True
                                )

    print(f"{Fore.CYAN}Starting optimization......{Style.RESET_ALL}")
    print(f"{Fore.CYAN}*{Style.RESET_ALL}" * 50)

    objective = objective_factory(X_train, y_train, X_val, y_val)
    study.optimize(objective,
                   n_trials=num_trails,
                   show_progress_bar=False)

    print(f"\n{Fore.CYAN}Optimization finished.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}*{Style.RESET_ALL}" * 50)
    report_study(study)

    return study

def report_study(study):
    print(f"{Fore.CYAN}Number of the total trials:{Style.RESET_ALL} {len(study.trials)}")

    finished_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"{Fore.CYAN}Number of finished  trials:{Style.RESET_ALL} {finished_trials}")

    pruned_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"{Fore.CYAN}Number of pruned trials:{Style.RESET_ALL} {pruned_trials}")

    print(f"{Fore.CYAN}Best trial after this run:{Style.RESET_ALL}")
    best_trial = study.best_trial

    print(f"{Fore.CYAN}  Trial number:{Style.RESET_ALL} {best_trial.number}")
    print(f"{Fore.CYAN}  Value (Best Validation Loss):{Style.RESET_ALL} {best_trial.value}")
    print(f"{Fore.CYAN}  Params:{Style.RESET_ALL} ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    # save the best trail paramaters and metrics to a CSV file
    df = pd.DataFrame(study.best_trial.params, index=[0])
    df.to_csv(f"{study_folder_name}/{study.study_name}_best_trial_params.csv")
