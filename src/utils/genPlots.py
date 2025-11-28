import matplotlib.pyplot as plt
import numpy as np
import os
import optuna
from collections import Counter
from plotly.subplots import make_subplots
import math
from colorama import Fore, Style
import plotly.graph_objects as go
from src.utils.helpers import sph2cart
from src.settings.settings import study_folder_name, study_name

import pandas as pd
from matplotlib.colors import ListedColormap, Normalize, LogNorm
from matplotlib.cm import get_cmap
from matplotlib import patches
from matplotlib.path import Path
#%% Plotting data distribution

def BoxPlot_data(X_values, X_raw ,y_values, y_raw,
                 scaling="PowerTransform", datatype = "Training"):

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.boxplot(X_raw)
    plt.title(f"Boxplot of Raw {datatype} features", fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Raw Values', fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.boxplot(X_values)
    plt.title(f"Boxplot of Scaled {datatype} Features", fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Scaled Values', fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.suptitle(f"Boxplots of {datatype} Features Before and After Scaling with {scaling}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.boxplot(y_raw)
    plt.title(f"Boxplot of Raw {datatype} Targets", fontsize=14, fontweight='bold')
    plt.xlabel('Targets', fontsize=12, fontweight='bold')
    plt.ylabel('Raw Values', fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.boxplot(y_values)
    plt.title(f"Boxplot of Scaled {datatype} Targets", fontsize=14, fontweight='bold')
    plt.xlabel('Targets', fontsize=12, fontweight='bold')
    plt.ylabel('Scaled Values', fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.suptitle(f"Boxplots of {datatype} Targets Before and After Scaling with {scaling}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig

def save_plot(fig, plot_name, experiment_name="default_experiment", base_output_dir="experiments"):
    """
    Saves a matplotlib figure to a specified plot directory within the project.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        plot_name (str): The desired filename for the plot (e.g., "learning_curve.png").
        experiment_name (str): The name of the experiment to categorize plots.
                               Plots will be saved in base_output_dir/experiment_name/plots/.
        base_output_dir (str): The base directory for all experiments.
    """
    # Defensive check to avoid AttributeError later
    if fig is None:
        raise ValueError("save_plot received fig=None. Make sure the plotting function returns a Figure.")

    # Construct the full path to the plots directory for this experiment
    if experiment_name == " ":
        plots_dir = base_output_dir
    else:
        plots_dir = os.path.join(base_output_dir, experiment_name)

    # Create the directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True) # exist_ok=True prevents error if dir already exists

    # Construct the full path for the plot file
    file_path = os.path.join(plots_dir, plot_name)

    # Save the figure
    fig.savefig(file_path)

    print(f"{Fore.RED}The plot: {Style.RESET_ALL}{plot_name} "
          f"{Fore.RED}saved to: {Style.RESET_ALL}{file_path}")

#%% visualize the results
def visualize_study(study):

    plot_dir = f"{study_folder_name}/{study.study_name}"
    # Create the directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)  # exist_ok=True prevents error if dir already exists

    fig = optuna.visualization.plot_optimization_history(study)
    # remove the title of the plot
    fig.update_layout(title=None)
    # place the legend inside the plot to the upside-right corner
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        xanchor="right",
        y=1,
        x=1
    ))
    fig.show()
    fig.write_image(f"{plot_dir}/optim_history.png", scale=3)

    fig = optuna.visualization.plot_param_importances(study)
    # remove the title of the plot
    fig.update_layout(title=None)
    # change the xlimit of the x-axis from 0 to 1
    fig.update_xaxes(range=[0, 1])
    # change the font of x-axis and y-axis labels
    fig.update_xaxes(title_font_size=14, tickfont_size=14)
    fig.update_yaxes(title_font_size=14, tickfont_size=14)
    # remove the tilte of the y-axis
    fig.update_yaxes(title=None)
    #making the font in bold for both x-axis and z-axis
    fig.update_layout(xaxis=dict(title_font_weight="bold", tickfont_weight="bold"),
                      yaxis=dict(title_font_weight="bold", tickfont_weight="bold"))
    # make the layout tight
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.show()
    fig.write_image(f"{plot_dir}/param_importance.png", scale=3)

    fig = optuna.visualization.plot_slice(study)
    fig.show()
    fig.write_image(f"{plot_dir}/slice_plot.png", scale=3)

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.update_layout(plot_bgcolor="white")  # "rgb(240, 248, 255)"
    fig.update_traces(line=dict(colorscale='Viridis', reversescale=False))
    fig.show()
    fig.write_image(f"{plot_dir}/parallel_coordinate.png", scale=3)

    fig = optuna.visualization.plot_timeline(study)
    fig.show()
    fig.write_image(f"{plot_dir}/timeline.png", scale=3)

    #fig = optuna.visualization.plot_intermediate_values(study)
    #fig.show()
    #fig.write_image(f"{plot_dir}/intermediate_values.png", scale=3)

def parallelCoordinatePlot(df, leading_column, colorbar=True, style="rainbow",
                           interpolation='linear', cross_val= False):

    print(Fore.RED + "Generating parallel coordinate plot..." + Style.RESET_ALL)

    # Validate the DataFrame contains the coloring column
    if leading_column not in df.columns:
        raise ValueError(f"Coloring column '{leading_column}' "
                         f"is not found in the DataFrame.")

    # Convert categorical columns to numeric codes for plotting
    cat_mappings = {}  # To store mappings for each categorical column
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    cat_cols = cat_cols.union(pd.Index(['bs']))
    numeric_df = df.copy()  # Make a copy of the DataFrame for numeric transformations
    # Convert categorical columns to numeric codes
    for col in cat_cols:
        if col in numeric_df.columns:
            # Transform categories to numeric codes
            numeric_df[col] = pd.Categorical(df[col]).codes
            # Store code-to-category mapping
            cat_mappings[col] = dict(enumerate(df[col].astype('category').cat.categories))

    plot_df = numeric_df.copy()  # Make a copy of the DataFrame for plotting

    # Drop the coloring column from the data frame if the colorbar is True
    if colorbar == True:
        plot_df = plot_df.drop(columns=[leading_column])
    else:
        # If the colorbar is False, we move the leading column to the last one in the dataframe
        plot_df = plot_df[[col for col in plot_df.columns if col != leading_column] + [leading_column]]

    # Drop the some column from the data frame if the colorbar is True
    plot_df = plot_df.drop(columns=["trial_number"])
    plot_df = plot_df.drop(columns=["R2"])
    plot_df = plot_df.drop(columns=["val_loss"])
    plot_df = plot_df.drop(columns=["Nparams"])
    plot_df = plot_df.drop(columns=["duration"])
    # Extract column names for plotting
    ynames = plot_df.columns
    ys = plot_df.values

    # Check for consistency in columns
    if ys.shape[1] != len(ynames):
        raise ValueError(
            f"Number of columns in the data ({ys.shape[1]}) does not match the number of column names ({len(ynames)})."
        )

    # Normalize data for parallel axis compatibility
    ymins = ys.min(axis=0)  # Minimum values of each column
    ymaxs = ys.max(axis=0)  # Maximum values of each column
    if cross_val:
        for i, col_name in enumerate(ynames):
            if "R2" in col_name:
                ymins[i] = 0.9
                ymaxs[i] = 1.0
            if "nMAE" in col_name:
                ymins[i] = 0.0
                ymaxs[i] = 12.5
            if "duration" in col_name:
                ymins[i] = 0
                ymaxs[i] = 200
            if "dr" in col_name:
                ymins[i] = 0.0
                ymaxs[i] = 0.5
            #if "BS" in col_name:
            #    ymins[i] = 32
            #    ymaxs[i] = 512
            if "Nparams" in col_name:
                ymins[i] = 0.1e6
                ymaxs[i] = 40e6
            if "nhl" in col_name:
                ymins[i] = 1
                ymaxs[i] = 10
            if "nhu" in col_name:
                ymins[i] = 32
                ymaxs[i] = 1024
            if "lr" in col_name:
                ymins[i] = 1e-5
                ymaxs[i] = 1e-3
            if "wd" in col_name:
                ymins[i] = 1e-6
                ymaxs[i] = 1e-3
    dys = ymaxs - ymins
    # Normalize values
    zs = np.zeros_like(ys) # Initialize the normalized data matrix
    zs[:, 0] = ys[:, 0] # Retain the first column as it is
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    # Initialize the plot
    fig, host = plt.subplots( figsize=(8, 5), nrows=1, ncols=1)
    axes = [host] + [host.twinx() for _ in range(ys.shape[1] - 1)] # creates a list of new Axes objects
    # Set logarithmic scale for lr column
        
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])      # Set y-axis limits for each axis
        ax.yaxis.set_ticks_position('left')  # Position the y-axis ticks on the left side
        # Set logarithmic scale for lr column
        if "lr" in ynames[i] or "wd" in ynames[i]:
            axes[i].set_yscale('log')
  
        # Format y-ticks
        if ymaxs[i] > 1000 and ymaxs[i] < 10000:
            ax.axes.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))  # Format y-ticks in thousands
        elif ymaxs[i] > 10000:
            ax.axes.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.0e}'))  # Format y-ticks in thousands
        elif ymins[i] < 1e-2 and ymaxs[i] <= 1e-1:
            ax.axes.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.0e}'))  # Format y-ticks in thousands
        # Handle categorical columns
        # If the column is categorical, set the y-ticks and labels accordingly
        col_name = ynames[i]
        if col_name in cat_mappings:
            reverse_mapping = {v: k for k, v in cat_mappings[col_name].items()}
            ax.set_yticks(list(cat_mappings[col_name].keys()))  # Set y-ticks for categorical data
            ax.set_yticklabels(list(reverse_mapping.keys()))  # Set y-tick labels for categorical data

        # Spines are the lines that frame the plot data area
        # (the top, bottom, left, and right borders).
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            # Position the right spine at a fraction of the axe's width
            ax.spines["right"].set_position(["axes", i / (ys.shape[1] - 1)])
            ax.spines["right"].set_linewidth(1.5)

    for ax in axes:
        # Set bold font weight for all y-tick labels
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')  # Make y-tick labels bold

        # Fine-tune tick parameters if necessary
        ax.tick_params(axis='y', labelsize=12, labelcolor='black', width=1.0)  # Adjust tick width/size

    # Configure axes labels
    host.set_xlim(0, ys.shape[1] - 1)  # Set x-axis limits to match the number of columns
    host.set_xticks(range(ys.shape[1]))  # Set x-ticks to match the number of columns
    host.set_xticklabels(ynames, fontsize=12,
                         fontweight='bold',
                         rotation=45,
                         ha='right')  # Rotate x-axis labels for better readability
    host.tick_params(axis='x', which='major', pad=15)  # Add padding to x-axis ticks
    host.spines['right'].set_visible(False)  # Hide the right spine
    host.xaxis.tick_bottom()  # Move x-axis ticks to the bottom

    # Plot the lines connecting points for each data row
    values_for_coloring = numeric_df[leading_column].to_numpy()

    if colorbar:
        if leading_column in cat_cols:
            # Handle categorical column with discrete colors
            unique_categories = np.unique(values_for_coloring)
            unique_categories.sort()
            # Create a colormap with distinct colors for each unique category
            cmap = ListedColormap(plt.cm.get_cmap(style, len(unique_categories)).colors)
            # Create a normalization object to map your data values to the colormap
            norm = Normalize(vmin=min(unique_categories), vmax=max(unique_categories), clip=True)
            # Create a mapping from unique categories to colors
            color_mapping = {category: color for category, color in zip(unique_categories, cmap.colors)}
            # Map the values to colors
            colors = [color_mapping[value] for value in values_for_coloring]
            # Create a ScalarMappable for the colorbar
            # This is necessary to create a colorbar with the correct colormap and normalization
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Create an empty array for the ScalarMappable
            cbar = plt.colorbar(sm, ax=axes[-1], orientation='vertical', pad=0.075)
            # cbar.set_ticks(list(color_mapping.keys()))  # Set colorbar ticks to the unique categories
            # cbar.set_ticklabels(list(color_mapping.keys()))  # Set colorbar tick labels to the unique categories
            cbar.set_ticks(range(len(unique_categories)))
            cbar.set_ticklabels(unique_categories)

        else:
            # Handle continuous column with normalized colors
            cmap = get_cmap(style)
            # Create a normalization object to map your data values to the colormap
            norm = plt.Normalize(vmin=values_for_coloring.min(), vmax=values_for_coloring.max())
            if cross_val:
                # logarithmic scale of the normalized data values
                #norm = LogNorm(vmin=1e0, vmax=1e1)
                norm = plt.Normalize(vmin=1e0, vmax=1e1)
            # Create a list of colors based on the colormap and normalized values
            colors = [cmap(norm(value)) for value in values_for_coloring]

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Create an empty array for the ScalarMappable
            cbar = plt.colorbar(sm, ax=axes[-1], orientation='vertical', pad=0.090)

        # Set the colormap for the ScalarMappable
        cbar.set_label(leading_column, fontsize=12, fontweight='bold')  # Set the colorbar label
        cbar.ax.yaxis.set_label_position(position="left")  # Position the colorbar label on the left

        cbar.ax.yaxis.set_tick_params(labelsize=12,  # Set the colorbar tick label size
                                      pad=5)  # Add padding to the colorbar ticks
        cbar.ax.yaxis.set_ticks_position('right')  # Position the colorbar ticks on the left
        cbar.outline.set_visible(False)  # Hide the colorbar outline
        # Set tick labels to bold
        for label in cbar.ax.yaxis.get_ticklabels():
            label.set_fontweight('bold')  # Make the ticks bold

    else:
        # Assign a default color if no colorbar is used
        # colors = ['blue'] * len(values_for_coloring)
        cmap = get_cmap(style)
        # Create a normalization object to map your data values to the colormap
        norm = plt.Normalize(vmin=values_for_coloring.min(), vmax=values_for_coloring.max())
        colors = [cmap(norm(value)) for value in values_for_coloring]

    # Plot the lines for each row in the DataFrame
    for j in range(ys.shape[0]):
        if interpolation == 'linear':
            verts = list(zip(np.linspace(0, ys.shape[1] - 1, len(ynames)), zs[j, :]))
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)

        elif interpolation == 'bezier':
            verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                             np.repeat(zs[j, :], 3)[1:-1]))
            # for x,y in verts: host.plot(x, y, 'go') to show the control points of the Béziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]

        path = Path(verts, codes)
        patch = patches.PathPatch(path,
                                  facecolor='none',  # No fill color
                                  lw=1.5,  # Line width
                                  edgecolor=colors[j],  # Use the color from the colormap
                                  alpha=0.4)  # Set transparency
        host.add_patch(patch)

    plt.tight_layout()
    plt.show()
    print(Fore.GREEN + "Parallel coordinate plot generated successfully!" + Style.RESET_ALL)
    # Save the figure if needed
    fig.savefig(f'{study_folder_name}/{study_name}/parallel_coordinate_plot.png',
                dpi=300, bbox_inches='tight')
    print(f"{Fore.RED}Figure saved at:{Style.RESET_ALL} "
          f"{study_folder_name}/{study_name}/parallel_coordinate_plot.png")

#%% plot the optimization history

def plot_trial_history(study: optuna.study, trial_number):
    """Plot training and validation history for a specific Optuna trial."""

    # Retrieve the history and params for the specified trial
    trial = study.trials[trial_number]
    history = trial.user_attrs.get("history", {})

    # Retrieve the Trial parameters
    params = trial.params

    if not history:
        print(f"No training history found for trial {trial_number}.")
        return

    # Plot the history using the existing function
    plot_hist(history, params=params)

def plot_study_history(study: optuna.study, start, end):
    plt.figure(figsize=(20, 12))
    # Plot validation loss for each trial
    val_loss, min_val_loss = [], []
    N = 0  # Counter for trials with no intermediate values
    for trial in study.trials[start:end]:
        # check if the trial has intermediate values
        if trial.intermediate_values is None or not trial.intermediate_values:
            print(f"\n{Fore.RED}Trial {trial.number} has no intermediate values.{Style.RESET_ALL}")
            N+= 1
        else:
            # Extract keys and values from intermediate_values
            x = list(trial.intermediate_values.keys())  # Steps or epochs (keys)
            y = list(trial.intermediate_values.values())  # Metrics (values)
            if y:  # Check if y has elements
                min_val_loss.append(min(y))
            else:
                # Fallback in case data is empty, append None or a placeholder
                min_val_loss.append(None)

            val_loss.append(trial.value)
            if trial.value is not None:
                print(
                      f"{Fore.BLUE}Trial: {Style.RESET_ALL}{trial.number} - "
                      f"{Fore.BLUE}duration: {Style.RESET_ALL}{trial.duration}s - "
                      f"{Fore.BLUE}Validation Loss:{Style.RESET_ALL} {trial.value:.4e} - "
                      f"{Fore.BLUE}min val_loss:{Style.RESET_ALL} {min(y):.4e} "
                      f"{Fore.BLUE}at Epoch:{Style.RESET_ALL} {np.argmin(y)}")
            else:
                if y:
                    print(
                        f"{Fore.BLUE}Trial: {Style.RESET_ALL}{trial.number} - "
                          f"{Fore.BLUE}Validation Loss:{Style.RESET_ALL} N/A "
                          f"{Fore.BLUE}min val_loss:{Style.RESET_ALL} {min(y):.4e} "
                          f"{Fore.BLUE}at Epoch:{Style.RESET_ALL} {np.argmin(y)} ")
                else:
                    print(
                        f"{Fore.BLUE}Trial: {Style.RESET_ALL}{trial.number} - "
                          f"{Fore.BLUE}Validation Loss:{Style.RESET_ALL} N/A "
                          f"{Fore.BLUE}min val_loss:{Style.RESET_ALL} N/A "
                          f"{Fore.BLUE}at Epoch:{Style.RESET_ALL} N/A ")

            plt.plot(x, y, label=f'Trial {trial.number}')
            plt.scatter(np.argmin(y), min(y), zorder=5)

    print(f"{Fore.RED}Minimum Validation Loss for min val_loss:"
          f"{Style.RESET_ALL}{min(min_val_loss):.4e} "
          f"{Fore.RED} for the Trial Number.:{Style.RESET_ALL} "
          f"{study.trials[start + N + min_val_loss.index(min(min_val_loss))].number}")
          #f"{min_val_loss.index(min(min_val_loss))}")
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=12, fontweight='bold')
    plt.title('Validation Loss for Each Trial', fontsize=16, fontweight='bold')
    # make the legend location to be the right of the plot
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
    plt.grid(True)
    plt.show()

def plot_param_frequencies(study, bins_for_float=20, max_unique=20, round_float=3):
    """
    Plot histograms showing how often each parameter value was chosen in an Optuna study.

    Args:
        study (optuna.Study): The Optuna study object.
        bins_for_float (int): Number of bins for continuous float parameters.
        max_unique (int): If a param has more unique values than this, it will be binned.
        round_float (int): Decimal precision for rounding float values.
    """
    # Collect all parameter names
    all_params = set()
    for t in study.trials:
        all_params.update(t.params.keys())

    num_params = len(all_params)
    cols = 3
    rows = math.ceil(num_params / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()

    for i, param in enumerate(sorted(all_params)):
        # Extract all values
        raw_values = [t.params.get(param) for t in study.trials if param in t.params]

        # Skip if no values
        if not raw_values:
            continue

        # Clean up None or invalid entries
        values = [v for v in raw_values if v is not None]

        # Handle numerical and categorical differently
        if all(isinstance(v, (int, float)) for v in values):
            unique_values = set(values)

            # Use binning if too many unique float values
            if len(unique_values) > max_unique:
                # Bin continuous values
                values = [round(v, round_float) for v in values]
                axs[i].hist(values, bins=bins_for_float, edgecolor='black')
            else:
                # Use frequency bar chart
                rounded = [round(v, round_float) for v in values]
                counts = Counter(rounded)
                axs[i].bar(counts.keys(), counts.values())
        else:
            # Treat as categorical
            string_values = [str(v) for v in values]
            counts = Counter(string_values)
            axs[i].bar(counts.keys(), counts.values())

        axs[i].set_title(param)
        axs[i].set_xlabel("Value")
        axs[i].set_ylabel("Frequency")
        axs[i].tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def plot_param_frequencies_interactive(study, max_unique=20, round_float=3, bins_for_float=20):
    """
    Plot interactive frequency histograms for all parameters in an Optuna study using Plotly.

    Args:
        study (optuna.Study): The Optuna study object.
        max_unique (int): If a parameter has more unique values than this, it will be binned.
        round_float (int): Rounding precision for float values.
        bins_for_float (int): Number of bins for continuous float values.
    """

    # Collect all parameter names
    all_params = set()
    for t in study.trials:
        all_params.update(t.params.keys())
    all_params = sorted(all_params)

    # Subplot layout
    cols = 2
    rows = math.ceil(len(all_params) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=all_params)

    param_axes = {
        "lr": "scientific",
        "weight_decay": "scientific",
        "batch size": "category"
    }

    for i, param in enumerate(all_params):
        # Collect values
        raw_values = [t.params.get(param) for t in study.trials if param in t.params]
        values = [v for v in raw_values if v is not None]

        row = i // cols + 1
        col = i % cols + 1

        # Handle numeric vs categorical
        if all(isinstance(v, (int, float)) for v in values):
            if param in param_axes and param_axes[param] == "scientific":
                # Manually bin and plot log-scale bar chart
                log_bins = np.logspace(np.log10(min(values)), np.log10(max(values)), bins_for_float)
                hist, bin_edges = np.histogram(values, bins=log_bins)
                bin_labels = [f"{bin_edges[i]:.1e}" for i in range(len(hist))]

                fig.add_trace(
                    go.Bar(x=bin_labels, y=hist, name=param),
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Value (log scale)", row=row, col=col)

            elif len(set(values)) > max_unique:
                # Bin general continuous numeric
                fig.add_trace(
                    go.Histogram(x=values, nbinsx=bins_for_float, name=param),
                    row=row, col=col
                )
            else:
                # Round and use bar
                rounded = [round(v, round_float) for v in values]
                counts = Counter(rounded)
                fig.add_trace(
                    go.Bar(x=list(counts.keys()), y=list(counts.values()), name=param),
                    row=row, col=col
                )
        else:
            # Categorical
            values = [str(v) for v in values]
            counts = Counter(values)
            fig.add_trace(
                go.Bar(x=list(counts.keys()), y=list(counts.values()), name=param),
                row=row, col=col
            )

    fig.update_layout(height=300 * rows, width=900, title_text="Parameter Frequency Histograms", showlegend=False)
    fig.show()

def plot_study_df(study_df):
    # color the dots by the trial number
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(study_df.trial_number, study_df.val_loss,
                c=study_df.duration, cmap='viridis')
    # show the color bar and grid
    plt.colorbar().set_label('Trial Duration (min)', weight='bold', size=12)
    plt.grid()
    # set the x-axis label and its ticks
    plt.xlabel('Trial Number', weight='bold', size=12)
    #plt.xticks(np.arange(0, max(study_df.trial_number) + 10, 10))
    # set the y-axis label
    plt.ylabel('Validation Loss', weight='bold', size=12)
    # set the title
    plt.title('Validation Loss vs. Trial Number', weight='bold', size=14)
    # draw a horizontal line at the minimum validation loss
    plt.axhline(y=min(study_df.val_loss), color='red',
                linestyle='--', label='Minimum Validation Loss')
    # show the plot
    plt.show()
    return fig
#%% Plot training loop history
def plot_hist(history, params=None):
    train_loss_hist = history.get('train_loss_hist', [])
    val_loss_hist = history.get('val_loss_hist', [])

    train_acc_hist = history.get('train_acc_hist', [])
    val_acc_hist = history.get('val_acc_hist', [])

    train_sMAPE_hist = history.get('train_sMAPE_hist', [])
    val_sMAPE_hist = history.get('val_sMAPE_hist', [])

    train_nMAE_hist = history.get('train_nMAE_hist', [])
    val_nMAE_hist = history.get('val_nMAE_hist', [])

    train_MSE_hist = history.get('train_MSE_hist', [])
    val_MSE_hist = history.get('val_MSE_hist', [])

    train_r2_hist = history.get('train_r2_hist', [])
    val_r2_hist = history.get('val_r2_hist', [])

    # Create a figure and a set of subplots (1 row, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()  # Flatten the grid for 1D indexing
    # Set a main title for the entire figure
    fig.suptitle(f"------------A DNN model with a {params.get("loss function")} loss function----------------\n "
                 f"#{history.get('num_params')} parameters "
                 f"\n{params.get('hidden_layers')} hidden layers,    "
                 f"{params.get('hidden_units')} units,    "
                 f"\n{params.get('width_type')} width shape,    "
                 f"{params.get('drop_out'):.3e} drop out rate,    "
                 f"\n{params.get('lr'):.3e} learning rate,    "
                 f"{params.get('weight decay'):.3e} weight decay,    "
                 f"\n__________and {params.get('batch size')} Batch Size__________",
                 fontsize=18, fontweight='bold', color='blue')

    # Plot train vs. validation loss
    axes[0].plot(train_loss_hist, label="Train Loss")
    axes[0].plot(val_loss_hist, label="Validation Loss")
    axes[0].scatter(np.argmin(val_loss_hist), min(val_loss_hist), color='red', zorder=5,
                    label=f'Min Val Loss: {min(val_loss_hist):.4e}')
    axes[0].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Loss", fontsize=12, fontweight='bold')
    axes[0].set_title("Train vs Validation Loss", fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right', frameon=True, shadow=True)
    axes[0].grid(True)

    # plot the accuracy of the model
    axes[1].plot(train_acc_hist, label="Train Accuracy")
    axes[1].plot(val_acc_hist, label="Validation Accuracy")
    axes[1].scatter(np.argmax(val_acc_hist), max(val_acc_hist), color='red', zorder=5,
                    label=f'Max Val Acc: {max(val_acc_hist):.4f}')
    axes[1].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    axes[1].set_title("Train vs Validation Accuracy", fontsize=16, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True)

    # plot the sMAPE of the model
    axes[2].plot(train_sMAPE_hist, label="Train sMAPE")
    axes[2].plot(val_sMAPE_hist, label="Validation sMAPE")
    axes[2].scatter(np.argmin(val_sMAPE_hist), min(val_sMAPE_hist), color='red', zorder=5,
                    label=f'Min Val Loss: {min(val_sMAPE_hist):.4f}')
    # Add a horizontal line at y=3
    axes[2].axhline(y=3, color='red', linestyle='--', label='Threshold')
    # Fill an area below the line with pale green
    axes[2].fill_between(range(len(train_sMAPE_hist)), 0, 3, color='palegreen', alpha=0.3)

    axes[2].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("sMAPE", fontsize=12, fontweight='bold')
    axes[2].set_title("Train vs Validation sMAPE", fontsize=16, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True)

    # plot the R2 of the model
    axes[3].plot(train_r2_hist, label="Train R2")
    axes[3].plot(val_r2_hist, label="Validation R2")
    axes[3].scatter(np.argmax(val_r2_hist), max(val_r2_hist), color='red', zorder=5,
                    label=f'Max Val R2: {max(val_r2_hist):.4f}')
    axes[3].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[3].set_ylabel("R2", fontsize=12, fontweight='bold')
    axes[3].set_title("Train vs Validation R2", fontsize=16, fontweight='bold')
    axes[3].legend()
    axes[3].grid(True)

    # plot the nMAE of the model
    axes[4].plot(train_nMAE_hist, label="Train nMAE")
    axes[4].plot(val_nMAE_hist, label="Validation nMAE")
    axes[4].scatter(np.argmin(val_nMAE_hist), min(val_nMAE_hist), color='red', zorder=5,
                    label=f'Min Val Loss: {min(val_nMAE_hist):.4f}')
    # Add a horizontal line at y=0.1
    axes[4].axhline(y=3, color='red', linestyle='--', label='Threshold')
    # Fill an area below the line with pale green
    axes[4].fill_between(range(len(train_nMAE_hist)), 0, 3, color='palegreen', alpha=0.3)

    axes[4].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[4].set_ylabel("nMAE", fontsize=12, fontweight='bold')
    axes[4].set_title("Train vs Validation nMAE", fontsize=16, fontweight='bold')
    axes[4].legend()
    axes[4].grid(True)

    # plot the MSE of the model
    axes[5].plot(train_MSE_hist, label="Train MSE")
    axes[5].plot(val_nMAE_hist, label="Validation MSE")

    if val_MSE_hist:
        min_idx = int(np.argmin(val_MSE_hist))
        min_val = float(min(val_MSE_hist))
        axes[5].scatter(min_idx, min_val, color='red', zorder=5,
                        label=f'Min Val MSE: {min_val:.4f}')

    # Add a horizontal line at y=0.1
    axes[5].axhline(y=3, color='red', linestyle='--', label='Threshold')
    # Fill an area below the line with pale green
    axes[5].fill_between(range(len(train_MSE_hist)), 0, 3, color='palegreen', alpha=0.3)

    axes[5].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[5].set_ylabel("MSE", fontsize=12, fontweight='bold')
    axes[5].set_title("Train vs Validation MSE", fontsize=16, fontweight='bold')
    axes[5].legend()
    axes[5].grid(True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    return fig

#%% plot True predicted
def plot_true_pred(results_df):
    mse = ((results_df['True_Values'] - results_df['Predicted_Values']) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = (results_df['True_Values'] - results_df['Predicted_Values']).abs().mean()
    mape = (np.abs(results_df['True_Values'] - results_df['Predicted_Values']) /
            np.abs(results_df['True_Values'])).mean() * 100
    from sklearn.metrics import r2_score
    r2 = r2_score(results_df['True_Values'], results_df['Predicted_Values'])
    results_df.head(10)
    print(f"\n{Fore.RED}Error Metrics:{Style.RESET_ALL}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2 Score: {r2:.4f}")

    fig = plt.figure(figsize=(10, 6))

    #plt.figure(figsize=(10, 6))
    plt.scatter(results_df['True_Values'], results_df['Predicted_Values'],
                alpha=0.5, label='True Values', color='skyblue')
    plt.plot([results_df['True_Values'].min(), results_df['True_Values'].max()],
             [results_df['True_Values'].min(), results_df['True_Values'].max()],
             'r--', lw=2, label='Predicted Values')
    plt.xlabel('True Values', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.legend()
    plt.title(f'True vs Predicted Values wit R2={r2:.4f}', fontsize=16, fontweight='bold')
    plt.grid(True, which="major", linestyle='--', linewidth=0.5)
    plt.show()

    return fig

#%% Plot 3D prediction

def plotScatter(zen,azi,r):
    """
    Plot a interactive 3D scatter plot of the data.
    :param zen: [rad] zenith angles
    :param azi: [rad] azimuth angles
    :param r: radius
    """

    # if zen is a (1000,) array the n=1 else n=zen.shape[1]
    n = r.shape[0]
    if len(r.shape) == 1:
        m = 1
        r = r.reshape((-1,1))
    else:
        m = r.shape[1]

    x = np.zeros((n,m))
    y = np.zeros((n,m))
    z = np.zeros((n,m))
    for i in range(m):
        x[:,i], y[:,i], z[:,i] = sph2cart(zen, azi, r[:,i])

    fig = go.Figure()
    for i in range(m):
        # add 3D scatter plot with label
        fig.add_trace(go.Scatter3d(x=x[:,i], y=y[:,i], z=z[:,i], mode='markers', marker=dict(size=2), name = str(i+1)))
    fig.show()

def plotScatterSurface(zenScatter, aziScatter, rScatter,
                       zenSurface, aziSurface, rSurface,
                       show=True, fn=''):
    # ensure inputs are in correct shape
    # give error if shape of zenScatter, aziScatter, rScatter is not (n,1)
    if zenScatter.ndim != 1 or aziScatter.ndim != 1 or rScatter.ndim != 1:
        raise ValueError("zenScatter, aziScatter, and rScatter must be 1D arrays.")
    #

    # create xyz data
    xScatter, yScatter, zScatter = sph2cart(zenScatter, aziScatter, rScatter)
    xSurface, ySurface, zSurface = sph2cart(zenSurface, aziSurface, rSurface)

    # create a figure
    fig = go.Figure()
    # add 3D scatter of the raw data
    fig.add_trace(go.Scatter3d(x=xScatter, y=yScatter, z=zScatter, mode='markers', marker=dict(size=2), name = 'true' ))
    # add 3D surface of the predicted data
    fig.add_trace(go.Surface(x=xSurface,y=ySurface,z=zSurface, surfacecolor= rSurface, colorscale='Viridis', showscale=True, colorbar=dict(title='pred')))
    # change layout
    fig.update_layout(
        #title='3D Scatter and Surface Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    if show:
        fig.show()
    if fn != '':
        fig.write_html(fn)

#%% comparitive plot
def Plot_feature_sets(iteration, trials: int):
    # read dataframe
    df1 = pd.read_csv(f'optuna_results/X1/X1Model_standardScaler{iteration}/study_results.csv')
    df2 = pd.read_csv(f'optuna_results/X2/X2Model_standardScaler{iteration}/study_results.csv')
    df3 = pd.read_csv(f'optuna_results/X3/X3Model_standardScaler{iteration}/study_results.csv')
    df4 = pd.read_csv(f'optuna_results/X4/X4Model_standardScaler{iteration}/study_results.csv')

    # sort dataframe by the best validation loss
    df1 = df1.sort_values(by='val_loss', ascending=True)
    df2 = df2.sort_values(by='val_loss', ascending=True)
    df3 = df3.sort_values(by='val_loss', ascending=True)
    df4 = df4.sort_values(by='val_loss', ascending=True)

    # Consider only the first N trials
    df1 = df1.head(trials)
    df2 = df2.head(trials)
    df3 = df3.head(trials)
    df4 = df4.head(trials)

    # Normalize marker sizes for better visualization
    all_params = pd.concat([df1.Nparams, df2.Nparams, df3.Nparams, df4.Nparams])
    min_params, max_params = all_params.min(), all_params.max()

    def normalize_size(params, min_size=50, max_size=200):
        """Normalize parameter counts to reasonable marker sizes"""
        if max_params == min_params:
            return np.full_like(params, (min_size + max_size) / 2)
        normalized = (params - min_params) / (max_params - min_params)
        return min_size + normalized * (max_size - min_size)

    # Plot the nMAE vs. duration for each trial
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.scatter(df1.duration, df1.nMAE, s=normalize_size(df1.Nparams),
                color='blue', alpha=0.6, edgecolors='darkblue', linewidth=1.5, label='X1')
    plt.scatter(df2.duration, df2.nMAE, s=normalize_size(df2.Nparams),
                color='red', alpha=0.6, edgecolors='darkred', linewidth=1.5, label='X2')
    plt.scatter(df3.duration, df3.nMAE, s=normalize_size(df3.Nparams),
                color='green', alpha=0.6, edgecolors='darkgreen', linewidth=1.5, label='X3')
    plt.scatter(df4.duration, df4.nMAE, s=normalize_size(df4.Nparams),
                color='purple', alpha=0.6, edgecolors='indigo', linewidth=1.5, label='X4')

    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Set labels with better formatting
    plt.xlabel('Duration [min]', weight='bold', size=14)
    plt.ylabel('nMAE [%]', weight='bold', size=14)
    #plt.title(f'nMAE vs. Duration (Top {trials} Trials)', weight='bold', size=16)

    # Find and mark the best performing trial across all models
    all_nMAE = pd.concat([df1.nMAE, df2.nMAE, df3.nMAE, df4.nMAE])
    min_nMAE = all_nMAE.min()
    plt.axhline(y=min_nMAE, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Best nMAE: {min_nMAE:.4f}')

    # Add legend with marker size reference
    first_legend = plt.legend(fontsize=14, framealpha=0.9, loc='upper right',
                              title='Feature sets', title_fontsize=16)

    # Create a second legend for marker sizes
    marker_sizes = [50, 100, 200]
    marker_labels = [f'{int(min_params):,}',
                     f'{int((min_params + max_params) / 2):,}',
                     f'{int(max_params):,}']

    legend_elements = [plt.scatter([], [], s=size, c='gray', alpha=0.6,
                                   edgecolors='black', linewidth=1.5,
                                   label=f'{label}')
                       for size, label in zip(marker_sizes, marker_labels)]

    second_legend = plt.legend(handles=legend_elements, loc='upper center',
                               fontsize=14, title=r'Model Size ($n_{params}$)',
                               framealpha=0.9, title_fontsize=16)
    # add legends to the figure
    ax.add_artist(first_legend)
    ax.axes.add_artist(second_legend)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    return fig