import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error, explained_variance_score, mean_absolute_percentage_error
import pandas as pd

def plot_model_residuals(y, y_pred, TARGET,
                         bins=100, binrange=None, kde=True, alpha=0.8, figsize=(14, 6),
                         xlim_hist=None, xlim_scatter=None, ylim_scatter=None):
    # --- Create plt Figure ---
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                        right=0.95, hspace=0.5, wspace=0.5)

    # --- Residual histplot ---
    # sns.histplot(data=(y - y_pred)/y, bins=bins, binrange=binrange,  kde=kde, stat = "probability", cumulative=True, ax=ax1)
    sns.histplot(data=(y - y_pred)/y, bins=bins, binrange=binrange, kde=kde,cumulative=False, ax=ax1)

    sim_val = (y - y_pred) / y
    result_df = pd.DataFrame({
        'Mean':np.mean(sim_val),
        'Standard_dev':np.std(sim_val),
        # 'Percentile(0.25)':np.percentile(sim_val, 25),
        # 'Median':np.percentile(sim_val, 50),
        # 'Percentile(0.75)':np.percentile(sim_val, 75),
        # print('R2 Score:',r2_score(y_test, y_pred))
        # print('RMSE:',mean_squared_error(y_test, y_pred, squared=False))
        # print('MAE:',mean_absolute_error(y_test, y_pred))
        # print("MAPE", mean_absolute_percentage_error(y_test, y_pred))
        'ERR < 5%' : sum([1 if (x>-0.05) & (x<0.05) else 0 for x in sim_val])/len(sim_val),
        'RMSE':mean_squared_error(y, y_pred, squared=False),
        'MAE':mean_absolute_error(y, y_pred),
        'R2 score':r2_score(y, y_pred)},index=['Results'])
    print(result_df)

    ax1.set_title(f'(Target set - Target predicted)', fontsize=14, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)
    ax1.set_xlabel(TARGET, fontsize=10, fontweight="bold")
    ax1.set_xlim(xlim_hist)
    ax1.grid(True)

    plt.figtext(0.15, 0.75, "Mean: %.2f" % 
                result_df.loc['Results','Mean'], fontsize=10, fontweight="bold")
    ax1.axvline(x=-0.05, linewidth=1, color='r')

    plt.figtext(0.15, 0.70, "Standard deviation: %.2f" %
                result_df.loc['Results','Standard_dev'], fontsize=10, fontweight="bold")
    ax1.axvline(x=np.percentile(sim_val, 50), linewidth=1, color='r')
    
    # plt.figtext(0.15, 0.65, "Percentile(0.25): %.2f" %
    #             result_df.loc['Results','Percentile(0.25)'], fontsize=10, fontweight="bold")
    ax1.axvline(x= 0.05, linewidth=1, color='r')

    
    # plt.figtext(0.15, 0.60, "Median: %.2f" %
    #             result_df.loc['Results','Median'], fontsize=10, fontweight="bold")
    # plt.figtext(0.15, 0.55, "Percentile(0.75): %.2f" % result_df.loc['Results','Percentile(0.75)'], fontsize=10, fontweight="bold")
    plt.figtext(0.15, 0.65, "ERR < 0.05: %.2f" %
                result_df.loc['Results','ERR < 5%'], fontsize=10, fontweight="bold")
    # --- Residual scatter ---
    ax2.scatter(y, y_pred, alpha=alpha)
    ax2.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)],
             linestyle='--', color='c')

    ax2.set_title(
        f'Actual and predicted values', fontsize=14, fontweight="bold")
    ax2.xaxis.set_tick_params(labelsize=10)
    ax2.yaxis.set_tick_params(labelsize=10)
    ax2.set_xlabel('Real ' + TARGET, fontsize=10, fontweight="bold")
    ax2.set_ylabel('Predicted ' + TARGET, fontsize=10, fontweight="bold")
    ax2.set_xlim(xlim_scatter)
    ax2.set_ylim(ylim_scatter)
    ax2.grid(True)

    # plt.figtext(0.60, 0.70, f"Mean absolute error: %.f" % result_df.loc['Results','MAE'], fontsize=11, fontweight="bold")
    plt.figtext(0.60, 0.75, f"Mean squared error: %.f" % result_df.loc['Results','RMSE'], fontsize=11, fontweight="bold")
    plt.figtext(0.60, 0.80, f"R2 score: %.2f" % result_df.loc['Results','R2 score'], fontsize=11, fontweight="bold")

    fig.tight_layout()
    plt.show()

