# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:40:43 2017

@author: georg
"""
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC



def create_classifier_pipeline_gs(classifier_type, my_dataset, my_features, 
                               parameters):
    
    data = featureFormat(my_dataset, my_features, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    
    # Using stratified shuffle split cross validation because of the small size of the dataset
    sss = StratifiedShuffleSplit(labels, 1000, random_state=42)

    scaler = MinMaxScaler()
    
    classifier = create_default_classifer(classifier_type)
    
    pipeline = Pipeline(steps=[('min_max_scaler', scaler), 
                               (classifier_type, classifier)])
    
    # Get optimized parameters for F1-scoring metrics
    grid_search = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=sss)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    
    grid_search.fit(features, labels)

    return grid_search


def create_default_classifer(x):
    # switch statement Python replacement - http://stackoverflow.com/a/103081
    return {
        'DT': DecisionTreeClassifier(random_state = 42),
        'KNN': KNeighborsClassifier(),
        #False Parameter given as  n_samples > n_features.
        "linear_SVC": LinearSVC(dual=False, random_state=42),
    }.get(x)
