#%% import necessary libraries
from src.settings.settings import device, train_filename
from src.data_preprocessing.preprocess import *
from src.models.dnn_model import LinearModel
import pandas as pd
import torch
#%% Loading the best model

def load_model(X_train, study, model_path, trial_number):
    # Create a new instance of the model with the same architecture
    # Check if trial_number is a number or dictionary

    if type(trial_number) == int:
        loaded_model = LinearModel(input_size=X_train.shape[1],
                                   hidden_units=study.trials[trial_number].params['hidden_units'],
                                   hidden_layer=study.trials[trial_number].params['hidden_layers'],
                                   width_type=study.trials[trial_number].params['width_type'],
                                   dropout_rate=study.trials[trial_number].params['drop_out']).to(device)

    if type(trial_number) == dict:
        loaded_model = LinearModel(input_size=X_train.shape[1],
                                   hidden_units=trial_number.get('hidden_units'),
                                   hidden_layer=trial_number.get('hidden_layers'),
                                   width_type=trial_number.get('width_type'),
                                   dropout_rate=trial_number.get('drop_out')).to(device)
    # Load the saved state_dict into the new model instance
    loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    loaded_model.eval()  # Set the model to evaluation mode
    print("\nLoaded Model State Dict (after loading saved weights):")
    for param_tensor in loaded_model.state_dict():
        print(param_tensor, "\t", loaded_model.state_dict()[param_tensor].size())

    return loaded_model

# %% Evaluate the model
def evaluate_model(loaded_model, y_test ,testData):
    scaler_X = testData.get('scaler_X', None)
    scaler_y = testData.get('scaler_y', None)
    X_test_raw = testData.get('X_test_raw', [])
    X_test_df = testData.get('X_test_df', [])
    y_test_df = testData.get('y_test_df', [])

    if scaler_X is None or scaler_y is None:
        raise ValueError("Both scaler_X and scaler_y must be provided in testData.")
    # Ensure X_test_raw is a NumPy array before scaler.transform
    X_test_raw_np = np.asarray(X_test_raw, dtype=np.float32)

    with torch.no_grad():
        X_sample = torch.tensor(scaler_X.transform(X_test_raw_np), dtype=torch.float32).to(device)
        y_pred_tensor = loaded_model(X_sample)
        # Convert predictions to CPU NumPy array (n_samples, 1)
        y_pred_np = np.asarray(y_pred_tensor.detach().cpu().view(-1).tolist(),
                               dtype=np.float32).reshape(-1, 1)


    # Ensure y_test is a CPU NumPy array (n_samples, 1)
    if isinstance(y_test, torch.Tensor):
        # Avoid Tensor.numpy() to bypass PyTorch's NumPy bridge
        y_test_np = np.asarray(y_test.detach().cpu().view(-1).tolist(),
                               dtype=np.float32).reshape(-1, 1)
    else:
        y_test_np = np.asarray(y_test,
                               dtype=np.float32).reshape(-1, 1)

    y_test_original = scaler_y.inverse_transform(y_test_np)
    y_pred_original = scaler_y.inverse_transform(y_pred_np)

    # Create a dataframe with true and predicted values
    results_df = pd.DataFrame()
    if X_test_df.shape[1] == 3 or "X3" in train_filename:
        results_df = pd.DataFrame({
            'zen': np.asarray(X_test_df['zen'].values),
            'azi': np.asarray(X_test_df['azi'].values),
            'uFac': np.asarray(X_test_df['uFac'].values),
            'F_1': np.asarray(y_test_df.values).flatten(),
            'y_pred_original': y_pred_original.flatten(),
            'y_test_original': y_test_original.flatten(),
            'True_Values': y_test_np.flatten(),
            'Predicted_Values': y_pred_np.flatten()
        })
    else:
        results_df = pd.DataFrame({
            'x': np.asarray(X_test_df['x'].values),
            'y': np.asarray(X_test_df['y'].values),
            'z': np.asarray(X_test_df['z'].values),
            'uFac': np.asarray(X_test_df['uFac'].values),
            'F_1': np.asarray(y_test_df.values).flatten(),
            'y_pred_original': y_pred_original.flatten(),
            'y_test_original': y_test_original.flatten(),
            'True_Values': y_test_np.flatten(),
            'Predicted_Values': y_pred_np.flatten()
        })

    # Format the values in scientific notation for better readability
    results_df['y_pred_original'] = results_df['y_pred_original'].apply(lambda x: f"{x:.6e}")
    results_df['y_test_original'] = results_df['y_test_original'].apply(lambda x: f"{x:.6e}")

    # Display the first few rows of the comparison
    print(f"\n{Fore.RED}Comparison of True vs Predicted Values:{Style.RESET_ALL}")
    pd.reset_option('display.max_columns')  # Reset to show all columns
    print(results_df.head(10))

    return results_df

