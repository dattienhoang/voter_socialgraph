# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:06:53 2017

@author: Dat Tien Hoang
"""

import networkx as nx
import numpy as np
import pandas as pd

path_dir = 'C:/Users/Dat Tien Hoang/Downloads/data_science_modeling_project/data_science_modeling_project/'

#------------------------------------------------------------------------------
# Part I - use graph data to make a new field reflecting network data
df1 = pd.read_csv(path_dir+'graph.csv')
# get all the unique id's from the graph data file
uniqid_G = pd.Series(df1['source'].unique()).append(pd.Series(df1['sink'].unique())).unique()

# make a graph, add nodes
G = nx.Graph()
for i in uniqid_G:
    G.add_node(i)
# now add the edges to the graph
print 'adding nodes and edges'
#for i in range(len(df1)):
#    #if i % 100 == 0:
#    #    print 'i= ', i, ' of ', range(len(df1))
#    #if df1.iloc[i]['source'] not in G.nodes():
#    #    G.add_node(df1.iloc[i]['source'])
#    #if df1.iloc[i]['sink'] not in G.nodes():
#    #    G.add_node(df1.iloc[i]['sink'])
#    G.add_edge(df1.iloc[i]['source'], df1.iloc[i]['sink'])
df1['source_sink'] = df1[['source', 'sink']].apply(tuple, axis=1)
G.add_edges_from(list(df1['source_sink']))

# do a quick  thing to find subgraphs
print 'making subgraphs'
sub_graphs = list(nx.connected_component_subgraphs(G))
#for i in range(len(sub_graphs)):
#    print "Subgraph:", i, "consists of ", len(sub_graphs[i].nodes()), '# of nodes'

# still need to assign new data to the dataframe with the rest of the data...later i guess

#------------------------------------------------------------------------------
# Part I - use graph data to make a new field reflecting network data
         
df2 = pd.read_csv(path_dir+'support.csv')
df3 = pd.read_csv(path_dir+'voters.csv')
df = pd.merge(df2, df3, on='prim_id')
df.head()

# add any new data fields
df['group'] = 0
prim_id = np.asarray(df['prim_id'])
group = np.asarray(df['group'])
print 'adding subgroup field'
for i in range(len(sub_graphs)):
    #for j in sub_graphs[i].nodes():
    #    df.loc[df['prim_id']==j,'group'] = i
    #df.loc[df['prim_id']==sub_graphs[i].nodes(),'group'] = i
    df.loc[df['prim_id'].isin(sub_graphs[i].nodes()),'group'] = i


# do any pre-processing needed here!
# first handle binary labels
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

print 'preprocess'
lb_list = ['demplus', 'repplus', 'ismarried', 'home', 'renters']
ohe_list = ['party', 'abrscore', 'hpt_nt', 'ethnic', 'sex', 'group']
#ohe = OneHotEncoder()
lb = LabelBinarizer()
#le = LabelEncoder()
for i in lb_list:
    df[i] = df[i].fillna(value='unk')
    df[i] = lb.fit_transform(df[i])

for i in ohe_list:
    df[i] = df[i].fillna(value='unk')
    #le_res = le.fit_transform(df[i])
    #df[i] = ohe.fit_transform(le_res.reshape(1,-1))
    df = df.join(pd.get_dummies(df[i], prefix = i+'_'))
    df = df.drop([i], axis=1)

# divide the data into test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['support'], axis=1), 
                                                    df['support'], test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_test)
print clf.score(X_test, y_test)

import xgboost as xgb
model  = xgb.XGBClassifier(max_depth=5)
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    #print model
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
#plt.figure()
#plt.title('XGBoost - Effect of n_estimators')
#plt.xlabel('n_estimator')
#plt.ylabel('r2 score')
#plt.ylim([0,1])
#plt.plot(estimators, scores)

#xgb.plot_importance(model, ylabel = list(df))

y_pred = model.predict(X_test)
predictions = [value for value in y_pred]
# evaluate predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#print 'xgboost done...', datetime.now()

