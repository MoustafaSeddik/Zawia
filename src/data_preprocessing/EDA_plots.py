import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from src.utils.genPlots import save_plot
from src.settings.settings import features_set

#%% Plot density distributions
def plot_density_distributions(X_train, X_val, X_test):
    fig = plt.figure(figsize=(12, 6))
    if X_train.shape[1] > 10:
        print("Warning: Plotting more than 10 features may take a long time.")
        return
    else:
        for i in range(X_train.shape[1]):
            plt.subplot(1, X_train.shape[1], i + 1)

            for X, label, color in [(X_train, 'Train', 'blue'),
                                    (X_val, 'Validation', 'green'),
                                    (X_test, 'Test', 'red')]:
                density = stats.gaussian_kde(X[:, i])
                xs = np.linspace(X[:, i].min(), X[:, i].max(), 200)
                ys = density(xs)
                plt.plot(xs, ys, color=color, label=label)
                plt.fill_between(xs, ys, alpha=0.2, color=color)

            plt.title(f'Feature {i + 1}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            if i == 0:
                plt.legend()

    plt.tight_layout()
    plt.show()
    save_plot(fig, plot_name=f"{features_set}_DesnsityPlot",
              experiment_name=f"{features_set}",
              base_output_dir=f"training_results")
    return fig
