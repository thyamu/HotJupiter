#%%
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Phi values vs kzz over different temperature

# Phi values VS Kzz
allData = pd.DataFrame()
for spread in ["50", "100", "250", "500", "1000"]: # spread

    data_dir = "/Users/hkim78/work/2020-hotJupiter/data/atmosphere-uncertainty/%sk_spread/"%spread

    for t in np.arange(400, 2100, 100): # mean temperature
        data0 = pd.read_csv(data_dir + 'kzz0_temp%d_spread%s.csv'%(t, spread))
        data1 = pd.read_csv(data_dir + 'kzz1_temp%d_spread%s.csv'%(t, spread))
        data2 = pd.read_csv(data_dir + 'kzz2_temp%d_spread%s.csv'%(t, spread))
        data3 = pd.read_csv(data_dir + 'kzz3_temp%d_spread%s.csv'%(t, spread))

        data0 = data0[['Phi distribution', 'kzz']]
        data1 = data1[['Phi distribution', 'kzz']]
        data2 = data2[['Phi distribution', 'kzz']]
        data3 = data3[['Phi distribution', 'kzz']]

        allData = pd.concat([allData, data0, data1, data2, data3], ignore_index=True)

        # Split into dependent and independent variables
        X = allData.iloc[:, :-1].values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(X)

        Y = allData.iloc[:, -1].values

    #plots for kzz vs phi
    plot_dir =  "/Users/hkim78/work/2020-hotJupiter/plot/atmosphere-uncertainty/phi_distribution/"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    g = sns.jointplot(x = 'Phi distribution', y = 'kzz', data= allData, kind="hex").set_axis_labels(xlabel='Phi', ylabel= 'kzz')
    txt = "spread %s"%spread
    g.fig.suptitle(txt)
    plt.setp(g.ax_joint.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(g.ax_joint.set_yticklabels(["", "0", "", "10^6", "",  "10^8", "", "10^10"]))
    plt.tight_layout()
    plt.savefig(plot_dir + "kzz_vs_phi_spread%s.png"%spread)
    plt.show()




