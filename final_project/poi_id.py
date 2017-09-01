#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from build_classifier import create_classifier_pipeline_gs, create_default_classifer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import feature_selection
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
            "poi",
            "deferral_payments",
            "deferred_income",
            "director_fees",
            "exercised_stock_options",
            "expenses",
            "from_messages",
            "from_poi_to_this_person",
            "from_this_person_to_poi",
            "long_term_incentive",
            "loan_advances",
            "other",
            "bonus",
            "restricted_stock",
            "restricted_stock_deferred",
            "salary",
            "shared_receipt_with_poi",
            "to_messages",
            "total_payments",
            "total_stock_value"
            ]


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

EXPLORE_DATA = True
if EXPLORE_DATA: 
    print "Amount of users: {}".format(len(data_dict))
    poi_amount = [p for p in data_dict if data_dict[p]["poi"] == True]
    print "Amount of POI: {}".format(len(poi_amount))
    print ""
    
    features_list_ = [feature for feature in data_dict.itervalues().next()]
    df = pd.DataFrame.from_dict(data_dict, orient = 'index')   
    df = df[features_list_]
    df = df.replace('NaN', np.nan)
    print "Not null values: "
    df.info()
    print ""
    
    print "Null values: "
    print df.isnull().sum()
    
    print ""
    print "Null values:"
    print df.notnull().sum().sum()

### Task 2: Remove outliers

def showplot(data, x, y, mark_poi = False):
    x_vals = [data[point][x] for point in data]
    y_vals = [data[point][y] for point in data]
    plt.scatter(x_vals, y_vals)
    
    if mark_poi:
        for point in data:
            if data[point]['poi']:
                plt.scatter(data[point][x], data[point][y], color = "r")
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

#Plot before removing outliers
#showplot(data_dict, "salary", "bonus")

data_dict.pop( "TOTAL", 0 )

#Plot after removing outliers
#showplot(data_dict, "salary", "bonus")


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

def compute_fraction(numerator, denominator):
    if numerator == 'NaN' or denominator == 'NaN':
        return 0
    else:
        return round(float(numerator)/float(denominator),2)

def add_fraction(data_dict, numerator, denominator, new_name):
    num = data_dict[numerator]
    denom = data_dict[denominator]
    data_dict[new_name] = compute_fraction(num, denom)
    return data_dict

#Add new features here
for emp in my_dataset:
    my_dataset[emp] = add_fraction(my_dataset[emp],
                      'from_this_person_to_poi',
                      'from_messages',
                      'fraction_to_poi')
    
    
    my_dataset[emp] = add_fraction(my_dataset[emp],
                      'from_poi_to_this_person', 
                      'to_messages',
                      'fraction_from_poi')
    
features_list.append('fraction_to_poi')
features_list.append('fraction_from_poi')

#Plot new features
#showplot(my_dataset, "fraction_from_poi","fraction_to_poi", mark_poi = True)



### Extract features and labels from dataset for k_best testing
k_best = feature_selection.get_k_best_features(my_dataset, features_list, 10)
my_features = ['poi'] + k_best

#Split data using newly selected features
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
#Create a basic test / train split
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


#Loop through classifications, create classifier and test it
algorithms = ["DT","KNN", "linear_SVC"]

TEST_BASIC_ALGORITHMS = False
if TEST_BASIC_ALGORITHMS:
    for algorithm in algorithms:
        print "Testing - {}".format(algorithm)
        clf = create_default_classifer(algorithm)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        acc_score = round(accuracy_score(labels_test, pred),2 )
        print "accuracy: {}".format(acc_score)
        print "recall: {}".format(recall_score(labels_test, pred))
        print "precision: {}".format(precision_score(labels_test, pred))
        print ""

        

### TUNE EACH CLASSIFIER PARAMETERS

TUNE_LINEAR_SVC = False
if TUNE_LINEAR_SVC:
    linear_parameters = dict(linear_SVC__tol = [0.0001, 0.001, 0.01, 0.1, 1],
                         linear_SVC__max_iter = [1000, 10000],
                         linear_SVC__C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    
    #Create GridSearchCV for Linear SVC
    cv_linear = create_classifier_pipeline_gs("linear_SVC", my_dataset,my_features, 
                                parameters = linear_parameters)
    
    linear_svc_best = cv_linear.best_estimator_
    print "Best parameters for linear SVC"
    print ""
    print cv_linear.best_params_
    test_classifier(linear_svc_best, my_dataset, my_features)


TUNE_KNN = False
if TUNE_KNN:
    knn_parameters = dict(KNN__weights = ['uniform', 'distance'],
                          KNN__n_neighbors = [3, 5, 15, 21],
                          KNN__p = [1, 2, 3],
                          KNN__leaf_size = [30, 50, 70, 100])
    
    cv_knn = create_classifier_pipeline_gs("KNN", my_dataset,my_features, 
                                parameters = knn_parameters)
    knn_best = cv_knn.best_estimator_
    print "Best parameters for KNN"
    print ""
    print cv_knn.best_params_
    test_classifier(knn_best, my_dataset, my_features)
    

TUNE_DT = False
if TUNE_DT:
    DT_parameters = dict(DT__min_samples_leaf=range(1, 5),
                          DT__max_depth=range(1, 5),
                          DT__class_weight=['balanced'],
                          DT__criterion=['gini', 'entropy'])
    
    cv_DT = create_classifier_pipeline_gs("DT", my_dataset, my_features, 
                                parameters = DT_parameters)
    
    dt_best = cv_DT.best_estimator_
    print "Best parameters for dt"
    print ""
    print cv_DT.best_params_
    test_classifier(dt_best, my_dataset, my_features)

### Final classifer


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


clf = Pipeline(steps=[('min_max_scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),  
                      ('DT', DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
            max_depth=2, max_features=None, max_leaf_nodes=None,
            min_samples_leaf=4, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))])
    
features_list = my_features

test_classifier(clf, my_dataset, features_list)
dump_classifier_and_data(clf, my_dataset, features_list)

