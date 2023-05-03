
## ML prediction efficacy of G, abundance, topology for disequilibrium based on

import os
import numpy as np
import pandas as pd
import json
import sklearn.model_selection as ms
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
import time



def XGB_accuracy(X, Y):
    """ Split for training and testing
    """
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=0)
    eval_set = [(x_train, y_train), (x_test, y_test)]

    """ Fit the decision tree
    """
    # classifier = xgb.XGBClassifier(objective="multi:softprob", min_child_wight=1000, max_depth=2, n_estimators=10000)
    classifier = xgb.XGBClassifier(objective="multi:softprob", min_child_wight=10, max_depth= 5, n_estimators=1000)
    # max_depth = 5 in the current draft
    classifier = classifier.fit(x_train, y_train, early_stopping_rounds=100, eval_set=eval_set,
                                eval_metric=["merror", "mlogloss"], verbose=False)
    """ Predictions
    """
    y_pred = classifier.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)


def remove_var(df, list_var):
    '''

    :param df: pandas dataframe
    :param list_var: list of column
    :return: df after removing columns in list_var, the list of variables
    '''
    return df.drop(list_var, axis=1)


### headers for different groups of features

header = ["Temperature", "Metallicity Altitude",
          "Mean Degree", "Average clustering coefficient",
          "Average node betweenness centrality", "Edge betweenness centrality",
          "Average shortest path length", "Average neighbor degree",
          "CO abundance", "CH4 abundance", "NH3 abundance",	"H2O abundance",
          "Delta G distribution",	"Phi distribution"
          "Mean of Temperature Distribution", "Kzz", "Spread of Uncertainty"]

header_abundance = ["CO abundance", "CH4 abundance", "NH3 abundance",	"H2O abundance"]

header_topology = ["Mean Degree", "Average clustering coefficient",
                "Average node betweenness centrality", "Edge betweenness centrality",
                "Average shortest path length", "Average neighbor degree"]


# compute accuracy of predicting Kzz using different combination of G, topoAve, and abundance.
dict_var = {
            #individual group
            'g':['Delta G distribution'],  'topo': header_topology, 'ab': header_abundance,
            #group combination
            'topo_ab': header_topology + header_abundance,
            'g_topo': ['Delta G distribution'] + header_topology,
            'g_ab' : ['Delta G distribution'] + header_abundance,
            #three group
            'g_topo_ab': ['Delta G distribution'] + header_topology + header_abundance,
            #individual topology
            'degree': ["Mean Degree"], 'cc': ["Average clustering coefficient"],
            'spl': ["Average shortest path length"],  'neighbor': ["Average neighbor degree"],
            'betw': ["Average node betweenness centrality"], 'edgebetw': ["Edge betweenness centrality"],
            #g + individual topology
            'g_degree': ['Delta G distribution'] + ["Mean Degree"],
            'g_cc': ['Delta G distribution'] + ["Average clustering coefficient"],
            'g_spl': ['Delta G distribution'] + ["Average shortest path length"],
            'g_neighbor': ['Delta G distribution'] + ["Average neighbor degree"],
            'g_betw': ['Delta G distribution'] + ["Average node betweenness centrality"],
            'g_edgebetw': ['Delta G distribution'] + ["Edge betweenness centrality"],
            #individual abundance
            'CO': ["CO abundance"], 'CH4': ["CH4 abundance"], 'NH3': ["NH3 abundance"], 'H2O': ["H2O abundance"],
            #c
            'g_CO': ['Delta G distribution'] + ["CO abundance"],
            'g_CH4': ['Delta G distribution'] + ["CH4 abundance"],
            'g_NH3': ['Delta G distribution'] + ["NH3 abundance"],
            'g_H2O': ['Delta G distribution'] + ["H2O abundance"],
            'spl_neighbor_betw_edgebetw': ["Average shortest path length"] + ["Average neighbor degree"]
                                  + ["Average node betweenness centrality"] + ["Edge betweenness centrality"],
            'g_spl_neighbor_betw_edgebetw': ['Delta G distribution']
                                    + ["Average shortest path length"] + ["Average neighbor degree"]
                                    + ["Average node betweenness centrality"] + ["Edge betweenness centrality"]
}

#header group
individual_group = ['g', 'topo', 'ab']
group_combination = ['topo_ab', 'g_topo', 'g_ab']
three_group = ['g_topo_ab']
simple_topo = ['degree', 'cc']
complex_topo = ['spl', 'neighbor']
betweenness = ['betw', 'edgebetw']
g_individual_topology = ['g_degree','g_cc', 'g_spl', 'g_neighbor', 'g_betw', 'g_edgebetw']
individual_abundance = ["CH4", "CO", "H2O", "NH3"]
g_individual_abundance = ['g_CO', 'g_CH4','g_NH3','g_H2O']


###
st = time.time()

data_dir = "/Users/hkim78/work/HotJupiter/data/perturbation/parsed_data/2021/"

list_temp = np.arange(400, 2100, 100) #==> due to lack of files
# list_temp = [400,  500,  600,  700,  800,  900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

for spread in ["500"]:
    for var in ["topo"]: # dict_var.keys(): #three_group: #dict_var.keys():
        st = time.time()
        print('starting', var)
        var_name = dict_var[var]

        dict_accuracy = dict()
        for removed_species in ["CH4", "CO", "NH3", "H2O"]:

            dict_accuracy[removed_species] = list()

            for t in list_temp: #np.arange(400, 2100, 100): ==>
                data0 = pd.read_csv(data_dir + '%s_removed/%s_removed_%sk_spread_kzz0-aveT%dK.csv' % (removed_species, removed_species, spread, t))
                data1 = pd.read_csv(data_dir + '%s_removed/%s_removed_%sk_spread_kzz1-aveT%dK.csv' % (removed_species, removed_species, spread, t))
                data2 = pd.read_csv(data_dir + '%s_removed/%s_removed_%sk_spread_kzz2-aveT%dK.csv' % (removed_species, removed_species, spread, t))
                data3 = pd.read_csv(data_dir + '%s_removed/%s_removed_%sk_spread_kzz3-aveT%dK.csv' % (removed_species, removed_species, spread, t))

                frames = [data0, data1, data2, data3]
                features = var_name + ['Kzz']
                allData = pd.concat(frames, ignore_index=True)

                allData = allData[features]

                """ Split into dependent and independent variables
                """
                X = allData.iloc[:, :-1]
                Y = allData.iloc[:, -1].values

                a = XGB_accuracy(X, Y)
                dict_accuracy[removed_species].append(a)

        result_dir = "/Users/hkim78/work/HotJupiter/ML/results/perturbed_data/2021/"
        output_path = result_dir + "perturbation_accuracy_spread%s_%s.json"%(spread, var)

        with open(output_path, 'w') as outfile:
            json.dump(dict_accuracy, outfile)

        et = time.time()

        print(var, (et - st))



