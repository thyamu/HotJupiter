
## ML prediction efficacy of G, abundance, topology for disequilibrium

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
    classifier = xgb.XGBClassifier(objective="multi:softprob", min_child_wight=10, max_depth= 5, n_estimators=1000) # current results in Hot Jupiter
    # classifier = xgb.XGBClassifier(objective="multi:softprob", min_child_wight=10, max_depth=3, n_estimators=500) # current test
    # max_depth = 5 in the current draft
    classifier = classifier.fit(x_train, y_train, early_stopping_rounds=100, eval_set=eval_set,
                                eval_metric=["merror", "mlogloss"], verbose=False)
    """ Predictions
    """
    y_pred = classifier.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)


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
            # individual abundance
            'CO': ["CO abundance"], 'CH4': ["CH4 abundance"], 'NH3': ["NH3 abundance"], 'H2O': ["H2O abundance"],
            # c
            'g_CO': ['Delta G distribution'] + ["CO abundance"],
            'g_CH4': ['Delta G distribution'] + ["CH4 abundance"],
            'g_NH3': ['Delta G distribution'] + ["NH3 abundance"],
            'g_H2O': ['Delta G distribution'] + ["H2O abundance"],
            'CO_NH3': ["CO abundance"] + ["NH3 abundance"],
            'spl_neighbor_betw_edgebetw': ["Average shortest path length"] + ["Average neighbor degree"]
                                          + ["Average node betweenness centrality"] + ["Edge betweenness centrality"],
            'g_CO_NH3': ['Delta G distribution'] + ["CO abundance"] + ["NH3 abundance"],
            'g_spl_neighbor_betw_edgebetw': ['Delta G distribution'] + ["Average shortest path length"] + ["Average neighbor degree"]
                                          + ["Average node betweenness centrality"] + ["Edge betweenness centrality"],
            'top_predictor': ['Delta G distribution']  + ["CO abundance"] + ["NH3 abundance"]
                             + ["Average shortest path length"] + ["Average neighbor degree"]
                             + ["Average node betweenness centrality"] + ["Edge betweenness centrality"],

}


individual_group = ['g', 'topo', 'ab']
group_combination = ['topo_ab', 'g_topo', 'g_ab']
three_group = ['g_topo_ab']
simple_topo = ['degree', 'cc']
complex_topo = ['spl', 'neighbor']
betweenness = ['betw', 'edgebetw']
g_individual_topology = ['g_degree','g_cc', 'g_spl', 'g_neighbor', 'g_betw', 'g_edgebetw']
individual_abundance = ["CH4", "CO", "H2O", "NH3"]
g_individual_abundance = ['g_CO', 'g_CH4','g_NH3','g_H2O']
individual_topology = simple_topo + complex_topo + betweenness
individual_features = ['g'] + individual_abundance + individual_topology

for var in ['top_predictor']: #dict_var.keys():
    st = time.time()
    print('starting', var)
    var_name = dict_var[var]

    dict_accuracy = dict()
    for spread in ["50", "250", "500", "1000"]: #spread
        data_dir = "/Users/hkim78/work/HotJupiter/data/atmosphere-uncertainty/parsed_data/2021/%sk_spread/"%spread
        dict_accuracy[spread] = list()

        for t in np.arange(400, 2100, 100):
            data0 = pd.read_csv(data_dir + 'kzz0-aveT%dK.csv'%(t))
            data1 = pd.read_csv(data_dir + 'kzz1-aveT%dK.csv'%(t))
            data2 = pd.read_csv(data_dir + 'kzz2-aveT%dK.csv'%(t))
            data3 = pd.read_csv(data_dir + 'kzz3-aveT%dK.csv'%(t))

            frames = [data0, data1, data2, data3]
            features = var_name + ['Kzz']
            allData = pd.concat(frames, ignore_index=True)

            allData = allData[features]

            """ Split into dependent and independent variables
            """
            X = allData.iloc[:, :-1]
            Y = allData.iloc[:, -1].values

            a = XGB_accuracy(X, Y)
            dict_accuracy[spread].append(a)

    result_dir = "/Users/hkim78/work/HotJupiter/ML/results/accuracy/2021/"
    output_path = result_dir + "accuracy_%s.json"%var

    with open(output_path, 'w') as outfile:
        json.dump(dict_accuracy, outfile)

    et = time.time()

    print(var, (et - st))



