import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import scipy.stats as stats
#%%
def plot_residual(results_df):
    r2 = r2_score(results_df['True_Values'], results_df['Predicted_Values'])
    results_df.head(10)
    residuals = results_df['True_Values'] - results_df['Predicted_Values']
    fig = plt.figure(figsize=(10, 6))

    plt.scatter(results_df['Predicted_Values'], residuals,
                alpha=0.7, label='residuals', color='skyblue', marker='o', edgecolors='blue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
    plt.xlabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12, fontweight='bold')
    plt.legend()
    plt.title(f'"Residual Plot" with R2={r2:.4f}', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()

    return fig

# plotting a histogram for residuals
def plot_residual_hist(results_df):
    residuals = results_df['True_Values'] - results_df['Predicted_Values']
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30,alpha=0.7, kde=True, color='skyblue')
    #plot a smooth line that follows the histogram
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Residuals (Actual - Predicted)", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.title("Histogram of Residuals", fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.show()
    return fig
# plotting normal Q-Q plot

def normalQQ_plot(results_df):
    residuals = results_df['True_Values'] - results_df['Predicted_Values']
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_xlabel("Theoretical Quantiles", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sample Quantiles (Residuals)", fontsize=12, fontweight='bold')
    ax.set_title("Normal Q-Q Plot", fontsize=16, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()
    return fig