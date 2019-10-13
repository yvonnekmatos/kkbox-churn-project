import pandas as pd
import numpy as np
from wd import *
from helpful_functions import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import RFE, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.tree import export_graphviz
import pydot
from ast import literal_eval
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# TRY DIFFERENT CLASSIFICATION MODELS AND CHOOSE OPTIMAL MODEL 

def get_metrics_and_roc_curve(model, model_name, X_test, y_test):
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:,1]

    roc_auc = np.round(roc_auc_score(y_test, pred_proba), decimals=6)
    print('***** {} *****'.format(model_name))
    print('accuracy: ', accuracy_score(y_test, pred))
    print('precision: ', precision_score(y_test, pred))
    print('recall: ', recall_score(y_test, pred))
    print('f1: ', f1_score(y_test, pred))
    print('roc_auc_score: ', roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)

    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.plot(fpr,tpr,label='{}, auc={}'.format(model_name, roc_auc))
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc=0)
    print('')

# Read in dataframe

for_modeling = pd.read_csv(data_directory+'rfe_reduced_feature_space.csv')

for_modeling.info()

list(for_modeling)

overlap = ['registered_via_woe'
           # , 'sum_num_unq_3last'
           , 'bd'
           # , 'sum_total_secs_clean_3last'
           # , 'reg_month_woe'
           # , 'num_unq_last'
           # , 'total_secs_clean_3rdlast'
           # , 'sum_num_25'
           # , 'reg_year'
           , 'city_woe'
           # , 'num_100_3rdlast'
           # , 'num_25_2ndlast'
           # , 'num_25_3rdlast'
           # , 'total_secs_clean_last'
           # , 'num_50_3rdlast'
           # , 'num_50_last'
           # , 'sum_num_75'
           # , 'sum_total_secs_clean'
           # , 'sum_num_100'
           # , 'sum_num_75_3last'
           # , 'sum_num_50'
           # , 'reg_day_of_week_woe'
           # , 'sum_num_985'
           ]


X = for_modeling[overlap]
y = for_modeling.is_churn_final


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=19, shuffle=True)

X_train, y_train = under_sample_data(X_train,y_train)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Could add CV for all of these in the future
knn = KNeighborsClassifier()
knn = knn.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train, y_train)


dt = DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(X_train, y_train)

nb = GaussianNB()
nb = nb.fit(X_train, y_train)

gbm = xgb.XGBClassifier()
gbm = gbm.fit(X_train, y_train)


models = {
            'knn': knn
          , 'log_reg': log_reg
          , 'DT': dt
          , 'RF': rf
          , 'naive_bayes': nb
          , 'xgb': gbm
          }

for k,v in models.items():
    get_metrics_and_roc_curve(v,k, X_test, y_test)


for k,v in models.items():
    get_metrics_and_roc_curve(v,k, X_train, y_train)

# Random forest model roc_auc_score = 0.70, recall = 0.61, precision = 0.22, accuracy = 0.67, f1 = 0.32
# XGB roc_auc_score = 0.73, recall = 0.66, precision = 0.22, accuracy = 0.67, f1 = 0.33
# Logistic regression roc_auc_score = 0.70, recall = 0.70, precision = 0.21, accuracy = 0.63, f1 = 0.32

#==============================================================================================================================
# PLOT TOP 3 MODELS

def get_metrics_and_roc_curve_specify_color(model, model_name, color, X_test, y_test):
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:,1]

    roc_auc = np.round(roc_auc_score(y_test, pred_proba), decimals=6)
    print('***** {} *****'.format(model_name))
    print('accuracy: ', accuracy_score(y_test, pred))
    print('precision: ', precision_score(y_test, pred))
    print('recall: ', recall_score(y_test, pred))
    print('f1: ', f1_score(y_test, pred))
    print('roc_auc_score: ', roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)

    plt.plot(fpr, tpr,lw=2, color=color)
    plt.plot([0,1],[0,1],c='#75bbfd',ls='--')
    # plt.plot(fpr,tpr,label='{}, auc={}'.format(model_name, roc_auc))
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # plt.legend(loc=0)
    plt.savefig(data_directory+'data_viz/roc_curve.png', transparent=True, dpi=1000)
    print('')

models = {
           'log_reg': [log_reg, '#1e488f']
          , 'RF': [rf, '#98eff9']
          , 'xgb': [gbm, '#5170d7']
          }


for k,v in models.items():
    get_metrics_and_roc_curve_specify_color(v[0],k, v[1], X_test, y_test)


for k,v in models.items():
    get_metrics_and_roc_curve_specify_color(v,k, X_train, y_train)
#==============================================================================================================================
# LOGISTIC REGRESSION MODEL

for_modeling = pd.read_csv(data_directory+'rfe_reduced_feature_space.csv')

for_modeling.info()


overlap = [
            'city_woe'
           # , 'gender_woe'
            , 'registered_via_woe'
           # , 'reg_month_woe'
           # , 'reg_day_of_week_woe'
           , 'bd'
           # , 'reg_year'
           # , 'sum_num_25'
           # , 'sum_num_985'
           # , 'sum_total_secs_clean'
           ]

X = for_modeling[overlap]
y = for_modeling.is_churn_final


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=19, shuffle=True)

X_train, y_train = under_sample_data(X_train,y_train)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train, y_train)
log_reg.coef_

get_metrics_and_roc_curve(log_reg, 'log_reg', X_train, y_train)
get_metrics_and_roc_curve(log_reg, 'log_reg', X_test, y_test)
