import copy
import optuna
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from time import time
from src.models.dnn_model import LinearModel
from src.settings.settings import *
from src.utils.genPlots import plot_hist, save_plot
#%% training function
# %% training function

def setup_training_components(X_train, params, optim, learning_rate, weight_decay, momentum, loss):
    """Setup model, criterion, and optimizer."""
    model = LinearModel(input_size=X_train.shape[1],
                        hidden_units=params.get('hidden_units'),
                        dropout_rate=params.get('drop_out'),
                        hidden_layer=params.get('hidden_layers'),
                        width_type=params.get('width_type')).to(device)

    modelParams = model.get_modelParameters()
    print(f"{Fore.RED}Total number of parameters in the model:{Style.RESET_ALL} {modelParams}")

    # Dictionary of available loss functions
    criterion_dict = {
        "smooth_l1": nn.SmoothL1Loss(),
        "MSE": nn.MSELoss(),
        "MAE": nn.L1Loss(),
        "huber": nn.HuberLoss()
    }

    try:
        criterion = criterion_dict[loss]
    except KeyError:
        raise ValueError(
            f"Unsupported loss function: {loss}. "
            f"Available methods: {list(criterion_dict.keys())}"
        )

    # Setup optimizer
    if optim == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum,
                                    nesterov=True)
    elif optim == "RMSprop":
        optimizer = torch.optim.RMSprop(params=model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=momentum)

    return model, criterion, optimizer, modelParams


def calculate_metrics(predictions, labels):
    """Calculate all metrics for a batch."""
    metrics = {}

    # Accuracy
    metrics['correct'] = ((predictions - labels).abs() < 0.1).sum().item()
    metrics['total'] = labels.size(0)

    # sMAPE
    metrics['sMAPE'] = torch.mean(
        torch.abs(predictions - labels) / (0.5 * (torch.abs(predictions) + torch.abs(labels)))
    ).item() * 100

    # nMAE
    mae = torch.mean(torch.abs(predictions - labels))
    metrics['nMAE'] = (mae / torch.mean(torch.abs(labels))).item() * 100

    # MSE
    metrics['MSE'] = nn.MSELoss()(predictions, labels).item()

    return metrics


def run_training_epoch(model, train_loader, criterion, optimizer, verbose_per_epoch, epoch, num_epochs):
    """Run one training epoch and return metrics."""
    model.train()
    running_loss = 0.0
    train_correct, train_total = 0, 0
    train_sMAPE, train_nMAE, train_MSE = 0.0, 0.0, 0.0
    train_preds, train_true = [], []

    with tqdm.tqdm(train_loader, unit="batch", mininterval=0,
                   disable=not verbose_per_epoch, ncols=105, colour="blue") as bar:
        bar.set_description(desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for batch_features, batch_labels in bar:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)

            # Calculate metrics
            metrics = calculate_metrics(predictions, batch_labels)
            train_correct += metrics['correct']
            train_total += metrics['total']
            train_sMAPE += metrics['sMAPE']
            train_nMAE += metrics['nMAE']
            train_MSE += metrics['MSE']

            # Store predictions for R2
            train_preds.extend(predictions.cpu().detach().tolist())
            train_true.extend(batch_labels.cpu().tolist())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=float(loss))

    # Compute average metrics
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    train_sMAPE = train_sMAPE / len(train_loader)
    train_nMAE = train_nMAE / len(train_loader)
    train_MSE = train_MSE / len(train_loader)
    train_r2 = r2_score(train_true, train_preds)

    return {
        'loss': train_loss,
        'acc': train_acc,
        'sMAPE': train_sMAPE,
        'nMAE': train_nMAE,
        'MSE': train_MSE,
        'r2': train_r2
    }


def run_validation_epoch(model, val_loader, criterion):
    """Run validation and return metrics."""
    model.eval()
    running_loss = 0.0
    val_correct, val_total = 0, 0
    val_sMAPE, val_nMAE, val_MSE = 0.0, 0.0, 0.0
    val_preds, val_true = [], []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            running_loss += loss.item()

            # Calculate metrics
            metrics = calculate_metrics(predictions, batch_labels)
            val_correct += metrics['correct']
            val_total += metrics['total']
            val_sMAPE += metrics['sMAPE']
            val_nMAE += metrics['nMAE']
            val_MSE += metrics['MSE']

            # Store predictions for R2
            val_preds.extend(predictions.cpu().tolist())
            val_true.extend(batch_labels.cpu().tolist())

    # Compute average metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_sMAPE = val_sMAPE / len(val_loader)
    val_nMAE = val_nMAE / len(val_loader)
    val_MSE = val_MSE / len(val_loader)
    val_r2 = r2_score(val_true, val_preds)

    return {
        'loss': val_loss,
        'acc': val_acc,
        'sMAPE': val_sMAPE,
        'nMAE': val_nMAE,
        'MSE': val_MSE,
        'r2': val_r2
    }


def update_history(history, train_metrics, val_metrics, epoch):
    """Update history dictionary with metrics from current epoch."""
    history["epoch"].append(epoch + 1)
    history["train_loss_hist"].append(train_metrics['loss'])
    history["train_acc_hist"].append(train_metrics['acc'])
    history["train_sMAPE_hist"].append(train_metrics['sMAPE'])
    history["train_nMAE_hist"].append(train_metrics['nMAE'])
    history["train_MSE_hist"].append(train_metrics['MSE'])
    history["train_r2_hist"].append(train_metrics['r2'])

    history["val_loss_hist"].append(val_metrics['loss'])
    history["val_acc_hist"].append(val_metrics['acc'])
    history["val_sMAPE_hist"].append(val_metrics['sMAPE'])
    history["val_nMAE_hist"].append(val_metrics['nMAE'])
    history["val_MSE_hist"].append(val_metrics['MSE'])
    history["val_r2_hist"].append(val_metrics['r2'])


def print_epoch_results(epoch, num_epochs, train_metrics, val_metrics, optimizer):
    """Print training and validation results for the epoch."""
    before_lr = optimizer.param_groups[0]["lr"]
    after_lr = optimizer.param_groups[0]["lr"]

    print(
        f"{Fore.BLUE}Epoch [{Style.RESET_ALL}{epoch + 1}/{num_epochs}]: "
        f"{Fore.BLUE}Train Loss: {Style.RESET_ALL}{train_metrics['loss']:.4e}, "
        f"{Fore.BLUE}Train Acc: {Style.RESET_ALL}{train_metrics['acc']:.2f}%, "
        f"{Fore.BLUE}Train sMAPE: {Style.RESET_ALL}{train_metrics['sMAPE']:.2f}%, "
        f"{Fore.BLUE}Train nMAE: {Style.RESET_ALL}{train_metrics['nMAE']:.2f}%, "
        f"{Fore.BLUE}Train R2: {Style.RESET_ALL}{train_metrics['r2']:.4f}, "
        f"{Fore.BLUE}initial lr: {Style.RESET_ALL}{before_lr:.2e}  "
    )
    print(
        f"{Fore.LIGHTYELLOW_EX}Epoch {Style.RESET_ALL}[{epoch + 1}/{num_epochs}]: "
        f"{Fore.LIGHTYELLOW_EX}Valid Loss: {Style.RESET_ALL}{val_metrics['loss']:.4e}, "
        f"{Fore.LIGHTYELLOW_EX}Valid Acc: {Style.RESET_ALL}{val_metrics['acc']:.2f}%, "
        f"{Fore.LIGHTYELLOW_EX}Valid sMAPE: {Style.RESET_ALL}{val_metrics['sMAPE']:.2f}%, "
        f"{Fore.LIGHTYELLOW_EX}Valid nMAE: {Style.RESET_ALL}{val_metrics['nMAE']:.2f}%, "
        f"{Fore.LIGHTYELLOW_EX}Valid R2: {Style.RESET_ALL}{val_metrics['r2']:.4f}, "
        f"{Fore.LIGHTYELLOW_EX}final lr: {Style.RESET_ALL}{after_lr:.2e} "
    )


def handle_optuna_trial(trial, val_metrics, history, epoch):
    """Handle Optuna trial reporting and pruning."""
    trial.report(val_metrics['loss'], epoch)
    trial.set_user_attr("history", history.copy())

    if trial.should_prune():
        print(f"{Fore.BLUE}=" * 50)
        print(f"{Fore.RED}Trial:{Style.RESET_ALL} {trial.number} "
              f"{Fore.RED}pruned at epoch:{Style.RESET_ALL} {epoch}. "
              f"{Fore.RED}Validation loss:{Style.RESET_ALL} {val_metrics['loss']:.4f}")
        print(f"{Fore.BLUE}=" * 50)
        raise optuna.exceptions.TrialPruned()


def model_train(X_train, y_train, x_val, y_val, params=None, trial=optuna.trial, optim="Adam"):
    """Main training function."""
    start_time = time()
    torch.manual_seed(42)

    # Extract parameters
    num_epochs = params.get('epochs', 1000)
    batch_size = params.get('batch size', 64)
    learning_rate = params.get('lr', 1e-5)
    weight_decay = params.get('weight decay', 0.0)
    momentum = params.get('momentum', 0.9)
    loss = params.get('loss function', "smooth_l1")

    # Create data loaders
    dataset_train = TensorDataset(X_train, y_train)
    dataset_val = TensorDataset(x_val, y_val)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    print(f"{Fore.RED}Length of train dataloader: {Style.RESET_ALL}{len(train_loader)} "
          f"{Fore.RED}batch size: {Style.RESET_ALL}{batch_size}")

    # Setup model, criterion, optimizer
    model, criterion, optimizer, modelParams = setup_training_components(
        X_train, params, optim, learning_rate, weight_decay, momentum, loss
    )

    # Initialize training state
    best_loss = float('inf')
    best_model = None
    history = {
        "epoch": [], "num_params": modelParams,
        "training_time": 0.0,
        "tm_2": [], "ep_2": [],
        "train_loss_hist": [], "val_loss_hist": [],
        "train_acc_hist": [], "val_acc_hist": [],
        "train_sMAPE_hist": [], "val_sMAPE_hist": [],
        "train_nMAE_hist": [], "val_nMAE_hist": [],
        "train_MSE_hist": [], "val_MSE_hist": [],
        "train_r2_hist": [], "val_r2_hist": []
    }

    verbose_per_epoch = trial is None

    # Training loop
    with tqdm.tqdm(range(num_epochs), unit="Epoch", mininterval=0,
                   disable=verbose_per_epoch, ncols=150, colour="blue") as trialBar:
        for epoch in trialBar:
            if trial is not None:
                trialBar.set_description(desc=f"Trial Number[{trial.number}]")

            # Run training and validation
            train_metrics = run_training_epoch(model, train_loader, criterion, optimizer,
                                               verbose_per_epoch, epoch, num_epochs)
            val_metrics = run_validation_epoch(model, val_loader, criterion)

            # Update history
            update_history(history, train_metrics, val_metrics, epoch)

            # Track time to reach nMAE <= 3.0
            if val_metrics['nMAE'] <= 3.0:
                tm_2 = (time() - start_time) / 60
                history["tm_2"].append(tm_2)
                history["ep_2"].append(epoch + 1)

            # Update progress bar
            trialBar.set_postfix({
                'val Loss': f"{val_metrics['loss']:.4e}",
                'Train Loss': f"{train_metrics['loss']:.4e}"
            })

            # Handle Optuna or print results
            if trial is not None:
                handle_optuna_trial(trial, val_metrics, history, epoch)
            else:
                print_epoch_results(epoch, num_epochs, train_metrics, val_metrics, optimizer)

            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_model = copy.deepcopy(model)

    # Finalize
    training_time = (time() - start_time) / 60
    history["training_time"] = training_time
    print(f"{Fore.BLUE}**************************************{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Total training time: {Style.RESET_ALL}{training_time:.2f} "
          f"{Fore.BLUE} minutes {Style.RESET_ALL}")
    print(f"{Fore.BLUE}**************************************{Style.RESET_ALL}")

    return history, model, best_model