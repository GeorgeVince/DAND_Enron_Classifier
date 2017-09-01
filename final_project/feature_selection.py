# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:40:17 2017

@author: georg
"""
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest


def get_k_best_features(data_dict, feature_list, num_features):
    data = featureFormat(data_dict, feature_list)
    target, features = targetFeatureSplit(data)
    
    clf = SelectKBest(k = num_features)
    clf = clf.fit(features, target)
    scores = clf.scores_
    #Sort features
    k_best = sorted(zip(feature_list[1:], scores), key = lambda x: x[1],
                 reverse = True)
    #Take n best features
    k_best = k_best[0:num_features]
    # Return only the feature names
    k_best = [k[0] for k in k_best]
    
    return k_best
    