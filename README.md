
# Zawia Deep Learning Pipeline

This project is designed to handle a full deep learning workflow, including data preprocessing, hyperparameter tuning with Optuna, training, and inference.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.12 installed and the required dependencies. You can install them using:
```shell script
pip install -r requirements.txt
```


### 2. Configuration
Before running the pipeline, you need to configure your experiment in the settings. You can find these parameters at the top of the script or within the `src/settings` directory (depending on your specific imports).

#### Key Parameters:
*   **`mode`**: Defines the operation to perform.
    *   `"training"`: Trains a model with current parameters.
    *   `"tuning"`: Runs Optuna hyperparameter optimization.
    *   `"predict"`: Runs inference on new data.
    *   `"analysis"`: Evaluates and visualizes results of comparative analysis.
*   **`features_set`**: Choose your input feature set (e.g., `"X1"`, `"X2"`, etc.).
*   **`normalization`**: Choose your scaling method (e.g., `"standardScaler"`, `"minMax"`, `"robustScaler"`).
*   **`iteration`**: Increment this number when running new tuning sessions to avoid overwriting previous databases.

### 3. Running the Pipeline
The entire workflow is orchestrated through the `MainPipeline.py` file located in the `src` directory.


## 📂 Project Structure
```shell script
Zawia/
├── data_training/          # Input data files (.npy format)
├── optuna_results/         # Hyperparameter tuning logs and databases
├── src/                    # Source code
│   ├── config/             # Configuration scripts
│   ├── data_preprocessing/ # Data cleaning and transformation logic
│   ├── inference/          # Scripts for model evaluation and prediction
│   ├── models/             # Model architecture definitions
│   ├── settings/           # Global parameters and experiment settings
│   ├── training/           # Model training loops and logic
│   ├── utils/              # Helper functions and utilities
│   └── MainPipeline.py     # Main entry point to run the project
├── .gitignore              # Files to be ignored by Git
├── LICENSE                 # Project license
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

## 📊 Results and Visualization
*   If `study_vis` is set to `True`, the pipeline will generate visualizations for the Optuna study.
*   If `predict_vis` is set to `True`, visualization of the prediction will be shown
