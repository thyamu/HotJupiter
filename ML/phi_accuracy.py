#%%
import os
import numpy as np
import pandas as pd
#import shap
import sklearn.model_selection as ms
from sklearn import preprocessing
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

#%%
### headers for different groups of features

header = ['Metallicity', 'Altitude',
        'Mean Degree',
        'CO Degree', 'CH4 Degree', 'NH3 Degree', 'H2O Degree',
        'Average shortest path length',
        'Average clustering coefficient',
        'CO clustering coefficient', 'CH4 clustering coefficient', 'NH3 clustering coefficient','H2O clustering coefficient',
        'CO node betweenness centrality', 'CH4 node betweenness centrality', 'NH3 node betweenness centrality',
        'H2O node betweenness centrality',
        'Edge betweenness centrality',
        'Average neighbor degree',
        'CO neighbor degree', 'CH4 neighbor degree', 'NH3 neighbor degree', 'H2O neighbor degree',
        'CO abundance', 'CH4 abundance', 'NH3 abundance', 'H2O abundance',
        'Delta G distribution', 'Phi distribution',
        'Average node betweenness centrality', 'Temperature', 'kzz']

header_average = [
    'Mean Degree', 'Average shortest path length', 'Average clustering coefficient',
    'Average neighbor degree','Average node betweenness centrality', 'Edge betweenness centrality']

header_abundance = [n for n in header if n.find('abundance') > -1] # 'CO abundance', 'CH4 abundance', 'NH3 abundance', 'H2O abundance']

header_CO = [n for n in header if n.find('CO') > -1]
header_CO_without_abundance = list(header_CO)
header_CO_without_abundance.remove('CO abundance')

header_CH4 = [n for n in header if n.find('CH4') > -1]
header_CH4_without_abundance = list(header_CH4)
header_CH4_without_abundance.remove('CH4 abundance')

header_NH3 = [n for n in header if n.find('NH3') > -1]
header_NH3_without_abundance = list(header_NH3)
header_NH3_without_abundance.remove('NH3 abundance')

header_H20 = [n for n in header if n.find('H2O') > -1]
header_H20_without_abundance = list(header_H20)
header_H20_without_abundance.remove('H2O abundance')

header_individual_cc = [n for n in header if n.find('clustering coefficient') > -1]
header_individual_cc.remove('Average clustering coefficient')

header_individual_betweenness = [n for n in header if n.find('node betweenness centrality') > -1]
header_individual_betweenness.remove('Average node betweenness centrality')

header_individual_degree = [n for n in header if n.find('Degree') > -1 and n.find('neighbor degree') == -1]
header_individual_degree.remove('Mean Degree')

header_individual_neighborDegree = [n for n in header if n.find('neighbor degree') > -1]
header_individual_neighborDegree.remove('Average neighbor degree')



#%%
def XGB_accuracy(X, Y):
    """ Split for training and testing
    """
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=0)
    eval_set = [(x_train, y_train), (x_test, y_test)]

    """ Fit the decision tree
    """
    # classifier = xgb.XGBClassifier(objective="multi:softprob", min_child_wight=1000, max_depth=2, n_estimators=10000)
    classifier = xgb.XGBClassifier(objective="multi:softprob", min_child_wight=10, max_depth=5, n_estimators=1000)
    classifier = classifier.fit(x_train, y_train, early_stopping_rounds=100, eval_set=eval_set,
                                eval_metric=["merror", "mlogloss"], verbose=False)
    """ Predictions
    """
    y_pred = classifier.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)

#%%
# Delta G distribution + Average Topology VS Kzz


dict_accuracy = dict()
for spread in ["50", "100", "250", "500", "1000"]: #spread

    data_dir = "/Users/hkim78/work/2020-hotJupiter/data/atmosphere-uncertainty/%sk_spread/"%spread
    plot_dir = "/Users/hkim78/work/2020-hotJupiter/plot/atmosphere-uncertainty/%sk_spread/"%spread

    dict_accuracy[spread] = list()
    for t in np.arange(400, 2100, 100):
        data0 = pd.read_csv(data_dir + 'kzz0_temp%d_spread%s.csv'%(t, spread))
        data1 = pd.read_csv(data_dir + 'kzz1_temp%d_spread%s.csv'%(t, spread))
        data2 = pd.read_csv(data_dir + 'kzz2_temp%d_spread%s.csv'%(t, spread))
        data3 = pd.read_csv(data_dir + 'kzz3_temp%d_spread%s.csv'%(t, spread))

        frames = [data0, data1, data2, data3]
        features = ['Delta G distribution'] + header_average + ['kzz']
        allData = pd.concat(frames, ignore_index=True)

        allData = allData[features]


        """ Split into dependent and independent variables
        """
        X = allData.iloc[:, :-1]
        Y = allData.iloc[:, -1].values

        a = XGB_accuracy(X, Y)
        dict_accuracy[spread].append(a)


#%%
dir_plot =  "/Users/hkim78/work/2020-hotJupiter/plot/atmosphere-uncertainty/machine_learning/"
if not os.path.exists(dir_plot):
    os.mkdir(dir_plot)

#%%
fig = plt.plot()
for spread in ["50", "100", "250", "500", "1000"]:
    plt.plot(dict_accuracy[spread], label="spread %s"%spread)
    plt.xticks(list(range(17)), np.arange(400, 2100, 100), rotation=30)
    plt.xlabel("Mean Temperature")
    plt.ylabel("Accuracy for Predicting Kzz")
    plt.title("Gibbs Free Energy + Average Topology")
plt.legend()
plt.tight_layout()
plt.savefig(dir_plot + "predicting_accuracy_gibbs_aveTopo.png")
plt.show()


#%%
# Delta G distribution VS Kzz


dict_accuracy = dict()
for spread in ["50", "100", "250", "500", "1000"]: #spread

    data_dir = "/Users/hkim78/work/2020-hotJupiter/data/atmosphere-uncertainty/%sk_spread/"%spread
    plot_dir = "/Users/hkim78/work/2020-hotJupiter/plot/atmosphere-uncertainty/%sk_spread/"%spread

    dict_accuracy[spread] = list()
    for t in np.arange(400, 2100, 100):
        data0 = pd.read_csv(data_dir + 'kzz0_temp%d_spread%s.csv'%(t, spread))
        data1 = pd.read_csv(data_dir + 'kzz1_temp%d_spread%s.csv'%(t, spread))
        data2 = pd.read_csv(data_dir + 'kzz2_temp%d_spread%s.csv'%(t, spread))
        data3 = pd.read_csv(data_dir + 'kzz3_temp%d_spread%s.csv'%(t, spread))

        frames = [data0, data1, data2, data3]
        #frames = [data2, data3]
        allData = pd.concat(frames, ignore_index=True)
        allData = allData[['Delta G distribution', 'kzz']]


        """ Split into dependent and independent variables
        """
        X = allData.iloc[:, :-1]
        Y = allData.iloc[:, -1].values

        a = XGB_accuracy(X, Y)
        dict_accuracy[spread].append(a)







