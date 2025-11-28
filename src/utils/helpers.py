import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import optuna

from src.settings.settings import *
#%%
def sph2cart(zen, azi, r):
    """
    Convert spherical coordinates to cartesian coordinates.
    :param r: radius
    :param zen: zenith angle in radians
    :param azi: azimuth angle in radians
    :return: x, y, z: cartesian coordinates
    """
    x = r * np.sin(zen) * np.cos(azi)
    y = r * np.sin(zen) * np.sin(azi)
    z = r * np.cos(zen)

    return x, y, z

def ScatterSurfacedata(filename, uFac, gridsize, loaded_model, testData):
    ############### Plot the raw data as scatter plot and the results data as surface ##################
    ## load true values
    df = filename
    """
        if os.path.exists(filename):
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        else:
            # Read npy train data from the provided filename and its corresponding y file
            X_train = np.load(filename, allow_pickle=True)
            train_dir = os.path.dirname(filename)
            y_train_filename = os.path.join(train_dir, "train_y.npy")
            y_train = np.load(y_train_filename, allow_pickle=True)

            X_train_df = pd.DataFrame(X_train)
            y_train_df = pd.DataFrame(y_train)

            # Build column names and assemble df for both 3 and 4 feature cases
            if X_train.shape[1] == 3 or ("X2" in filename):
                # Expected order: zen, azi, uFac
                columns = ["zen", "azi", "uFac"]
                X_train_df.columns = columns
                y_train_df.columns = ["F_1"]
                df = pd.concat([X_train_df, y_train_df], axis=1)
            elif X_train.shape[1] == 4 or ("X4" in filename):
                # Expected order: x, y, z, uFac (if your pipeline needs these mapped to zen/azi, adapt here)
                columns = ["x", "y", "z", "uFac"]
                X_train_df.columns = columns
                y_train_df.columns = ["F_1"]
                df = pd.concat([X_train_df, y_train_df], axis=1)
            else:
                raise ValueError(f"Unsupported feature count in {filename}: {X_train.shape[1]}")
    else:
        raise FileNotFoundError(f"File not found: {filename}. Please ensure the file exists at the specified location.")
    """


    # Filter by uFac safely
    if 'uFac' not in df.columns:
        raise KeyError("Expected column 'uFac' not found in input data.")
    df = df[df['uFac'] == uFac]

    # Determine scatter angle columns
    if {'zen', 'azi'}.issubset(df.columns):
        zenScatter = df['zen'].values
        aziScatter = df['azi'].values
    elif {'x', 'y'}.issubset(df.columns):
        # If only Cartesian available, convert or reuse as needed; placeholder uses x->zen, y->azi
        zenScatter = df['x'].values
        aziScatter = df['y'].values
    else:
        raise KeyError("Could not find angle columns (zen/azi or x/y) in data.")

    if 'F_1' not in df.columns:
        raise KeyError("Expected target column 'F_1' not found in input data.")
    rScatter = df['F_1'].values

    ## initialize surface grid
    zenSurface = np.radians(np.linspace(0, 90, gridsize))
    aziSurface = np.radians(np.linspace(0, 360, gridsize))
    # create meshgrid for surface
    zenSurface, aziSurface = np.meshgrid(zenSurface, aziSurface)

    # create a dataframe with zen, azi, and uFac
    X = np.column_stack((
        zenSurface.flatten(),
        aziSurface.flatten(),
        np.ones_like(zenSurface.flatten()) * uFac
    ))

    ## load model
    loaded_model = loaded_model.to(device)
    ##################
    scaler_X = testData.get('scaler_X', None)
    scaler_y = testData.get('scaler_y', None)

    if scaler_X is None or scaler_y is None:
        raise ValueError("Missing scalers in testData. "
                         "Ensure you pass testData from prepare_data().")

    ## evaluate model on defined incidence angles
    ##################
    with torch.no_grad():
        X_sample = torch.tensor(scaler_X.transform(X), dtype=torch.float32).to(device)
        rSurface = loaded_model(X_sample)
        rSurface = np.asarray(rSurface.detach().cpu().view(-1).tolist(),
                              dtype=np.float32).reshape(-1, 1)
        # retransform
        rSurface = scaler_y.inverse_transform(rSurface)
    ##################
    # define feature Matrix X
    # rSurface = model.predict(X)
    rSurface = np.reshape(rSurface, zenSurface.shape)

    return zenScatter, aziScatter, rScatter, zenSurface, aziSurface, rSurface

def setget_model_path(trial_number, itr = "Adam"):
    # Save the model of this trial
    model_dir = f"{trials_dir}/{trial_number}"
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
    model_filename = f"{itr}_bestModel_trial_{trial_number}.pth"
    trial_model_path = os.path.join(model_dir, model_filename)
    return trial_model_path

def study_results(study):
    print(f"{Fore.CYAN}Study statistics:{Style.RESET_ALL}")
    print(f"{Fore.RED}  Number of finished trials:{Style.RESET_ALL} {len(study.trials)}")
    print(f"{Fore.RED}  Number of pruned trials:{Style.RESET_ALL} "
          f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"{Fore.RED}  Number of complete trials:{Style.RESET_ALL} "
          f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    print(f"{Fore.CYAN}Best trial:{Style.RESET_ALL}")
    trial = study.best_trial

    print(f"{Fore.RED}  Value:{Style.RESET_ALL} {trial.value}")

    trial_number, duration, val_loss = [], [], []
    lr, BS, wd, nhu, nhl, dr, wtype = [],[],[],[],[],[], []
    Nparams, nMAE, R2= [],[],[]


    for trial in study.trials[:]:
        if trial.state == optuna.trial.TrialState.COMPLETE or trial.state == optuna.trial.TrialState.PRUNED:
            trial_number.append(trial.number)
            duration.append(trial.duration.total_seconds()/60 if trial.duration is not None else None)
            val_loss.append(trial.value)
            lr.append(trial.params['lr'])
            BS.append(trial.params['batch size'])
            wd.append(trial.params['weight decay'])
            nhu.append(trial.params['hidden_units'])
            nhl.append(trial.params['hidden_layers'])
            dr.append(trial.params['drop_out'])
            wtype.append(trial.params['width_type'])

            history = trial.user_attrs.get("history", {})
            Nparams.append(history.get("num_params", None))
            nMAE.append(min(history.get('val_nMAE_hist', None)))
            R2.append(max(history.get('val_r2_hist', None)))


            #Nparams.append(trial.user_attrs.get("history", {"num_params"}))
            #nMAE.append(min(trial.user_attrs.get('history', {"nMAE"})))
            #R2.append(max(trial.user_attrs.get('history', {"R2"})))
            #tt.append(trial.user_attrs.get('history', {"training_time"}))

    # create a dataframe for the results
    df = pd.DataFrame({
        'st': wtype,
        'trial_number': trial_number,
        'Nparams': Nparams,
        'nhu': nhu,
        'nhl': nhl,
        'lr': lr,
        'bs': BS,
        'wd': wd,
        'do': dr,
        'nMAE': nMAE,
        'R2': R2,
        'val_loss': val_loss,
        'duration': duration,
    })

    print(df.head(len(trial_number)).sort_values(by='val_loss', ascending=True))
    # Save the results to a CSV file
    results_filename = f"study_results.csv"
    results_dir = f"{study_folder_name}/{study_name}"
    # Ensure the directory exists before saving
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, results_filename)
    df.to_csv(results_path, index=True)
    print(f"{Fore.RED}Results df saved to:{Style.RESET_ALL} {results_path}")
    return df



