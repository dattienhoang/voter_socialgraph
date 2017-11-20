# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:06:53 2017

@author: Dat Tien Hoang
"""

import networkx as nx
import numpy as np
import pandas as pd

#path_dir = 'C:/Users/Dat Tien Hoang/Downloads/data_science_modeling_project/data_science_modeling_project/'
path_dir = 'C:/Users/dhoang/Documents/GitHub/voter_socialgraph/'

#------------------------------------------------------------------------------
# Part I - use graph data to make a new field reflecting network/groups

# load the data
df1 = pd.read_csv(path_dir+'graph.csv')
# get all the unique id's from the graph data file
uniqid_G = pd.Series(df1['source'].unique()).append(pd.Series(df1['sink'].unique())).unique()

# make a graph, add nodes
G = nx.Graph()
for i in uniqid_G:
    G.add_node(i)
# now add the edges to the graph
print 'adding nodes and edges'
df1['source_sink'] = df1[['source', 'sink']].apply(tuple, axis=1)
G.add_edges_from(list(df1['source_sink']))

# do a quick  thing to find subgraphs
print 'making subgraphs'
sub_graphs = list(nx.connected_component_subgraphs(G))
# this part is really slow!
#subgroup_sizes = [len(x) for x in sub_graphs.nodes()]
#import matplotlib.pyplot as plt
#plt.hist(subgroup_sizes, bins='auto')  # arguments are passed to np.histogram
#plt.title("Distribution of Subgroup Sizes")
#plt.xlim(0,1)
#plt.show()
#for i in range(len(sub_graphs)):
#    print "Subgraph:", i, "consists of ", len(sub_graphs[i].nodes()), '# of nodes'

#------------------------------------------------------------------------------
# Part II - merge data and add new field about subgroups
         
# load more data
df2 = pd.read_csv(path_dir+'support.csv')
df3 = pd.read_csv(path_dir+'voters.csv')
df = pd.merge(df2, df3, on='prim_id')
print df.head()

# add new data fields
df['group'] = 0
prim_id = np.asarray(df['prim_id'])
group = np.asarray(df['group'])
print 'adding subgroup field'
for i in range(len(sub_graphs)):
    df.loc[df['prim_id'].isin(sub_graphs[i].nodes()),'group'] = i

#------------------------------------------------------------------------------
# Part II - merge data and add new field about subgroups

# do any pre-processing needed here!
# first handle binary labels
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

print 'preprocess'
lb_list = ['demplus', 'repplus', 'ismarried', 'home', 'renters']
ohe_list = ['party', 'abrscore', 'hpt_nt', 'ethnic', 'sex', 'group']
lb = LabelBinarizer()
for i in lb_list:
    df[i] = df[i].fillna(value='unk')
    df[i] = lb.fit_transform(df[i])
for i in ohe_list:
    df[i] = df[i].fillna(value='unk')
    df = df.join(pd.get_dummies(df[i], prefix = i+'_'))
    df = df.drop([i], axis=1)

#------------------------------------------------------------------------------
# Part III - run the model on training and test sets

# divide the data into test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['support', 'prim_id'], axis=1), 
                                                    df['support'], test_size=0.33, random_state=42)

def RF_model(X, y, note, model):
    from sklearn.ensemble import RandomForestClassifier
    if model == 0:
        clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=50)
    else:
        clf = model
    clf.fit(X, y)
    clf.predict(X)
    print "Features sorted by their score:"
    names = df.drop(['support'], axis=1).columns
    print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), 
                 reverse=True)
    print "Accuracy: %.2f%%" % (clf.score(X_train, y_train) * 100.0)
    predicted_probabilities = clf.predict_proba(df.drop(['support', 'prim_id'], axis=1))

    import matplotlib.pyplot as plt
    plt.hist(predicted_probabilities[:,0], bins='auto')  # arguments are passed to np.histogram
    plt.title("Probability of Candid. A Vote: " + note)
    plt.xlim(0,1)
    plt.show()
    return clf, predicted_probabilities

model, res = RF_model(X_train, y_train, 'training', 0)
model, res = RF_model(X_test, y_test, 'testing', model)
model, res = RF_model(df.drop(['support', 'prim_id'], axis=1), df['support'], 'all', model)

#------------------------------------------------------------------------------
# Part IV - now run the model on all data, regardless if we have voter information on them

#df3 = pd.read_csv(path_dir+'voters.csv')
#print df3.head()

# add any new data fields
df3['group'] = 0
prim_id = np.asarray(df3['prim_id'])
group = np.asarray(df3['group'])
print 'adding subgroup field'
for i in range(len(sub_graphs)):
    df3.loc[df3['prim_id'].isin(sub_graphs[i].nodes()),'group'] = i
model, res = RF_model(df3.drop(['support', 'prim_id'], axis=1), df3['support'], 'all', model)

#import xgboost as xgb
#model  = xgb.XGBClassifier(max_depth=5)
#estimators = np.arange(10, 200, 10)
#scores = []
#for n in estimators:
#    model.set_params(n_estimators=n)
#    #print model
#    model.fit(X_train, y_train)
#    scores.append(model.score(X_test, y_test))
##plt.figure()
##plt.title('XGBoost - Effect of n_estimators')
##plt.xlabel('n_estimator')
##plt.ylabel('r2 score')
##plt.ylim([0,1])
##plt.plot(estimators, scores)
#
##xgb.plot_importance(model, ylabel = list(df))
#
#y_pred = model.predict(X_test)
#predictions = [value for value in y_pred]
## evaluate predictions
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
#
##print 'xgboost done...', datetime.now()

