import torch as torch
from colorama import Fore, Style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{Fore.RED}Using device: {Style.RESET_ALL}{device}")
print(f"{Fore.RED}CUDA available: {Style.RESET_ALL}{torch.cuda.is_available()}")
print(f"{Fore.RED}CUDA version: {Style.RESET_ALL}{torch.version.cuda}")
print(f"{Fore.RED}PyTorch version: {Style.RESET_ALL}{torch.__version__}")

#%% Settings parameters
mode = "analysis"   # "training" or "tuning", "predict", "explore", "analysis"
study_vis = True
predict_vis = True
predict = True

features_set = "X4"  # "F-din" or "X1"  "X2", "X3" or "X4"
normalization = "standardScaler" # "minMax", "standardScaler", "robustScaler", "powerTransformer""

train_filename = f"data_training/{features_set}.npy"
target_filename = f"data_training/y.npy"

iteration = 4 #  "_1" the number of times we tuned the hyperparameters for a specific feature set & normalization
study_name = f"{features_set}Model_{normalization}_{iteration}"  # 'F-dinModel_optimization' , f"F-dinModel_{normalization}"
database_name = f"{normalization}_{features_set}_study_{iteration}.db"  #  or "my_regression_study.db" for older studies, f"{normalization}_study.db"
study_folder_name = f"optuna_results/{features_set}"

storage_url = f"sqlite:///{study_folder_name}/{database_name}"
trials_dir = f"{study_folder_name}/{study_name}/trials"

trainings_folder_name = "training_results"
best_model_name = f"bestModel_{normalization}.pth"
best_model_path = f"{trainings_folder_name}/{features_set}/{best_model_name}"
