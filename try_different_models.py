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
    # plt.figure();
    plt.savefig(data_directory+'data_viz/roc_curve.png', transparent=True)
    print('')

# Read in dataframe

for_modeling = pd.read_csv(data_directory+'clean_transformed_data_for_simple_model2.csv')
# for_modeling = pd.read_csv(data_directory+'subset_model_w_listeing.csv')

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

# Random forest baseline model roc_auc_score = 0.70, recall = 0.61, precision = 0.22, accuracy = 0.67, f1 = 0.32
# WOE doesn't work well with RF - can lead to overfitting. For RF, it doesn't matter if features are independent
# XGB roc_auc_score = 0.73, recall = 0.66, precision = 0.22, accuracy = 0.67, f1 = 0.33
# Logistic regression roc_auc_score = 0.70, recall = 0.70, precision = 0.21, accuracy = 0.63, f1 = 0.32

X_train_names = np.array(list(X))
feature_importance = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), X_train_names), reverse=True)
feature_importance

gbm.feature_importances_
#================================================================================================================================
# TUNING XGBOOST 
param_grid = {'max_depth': [5, 7, 9, 11]
              , 'learning_rate': [0.05, 0.1, 0.5, 1]
              , 'subsample': [0.05, 0.1, 0.5, 1]
              , 'min_child_weight': [1, 100, 1000]
              , 'colsample_bytree': [0.1, 0.5, 0.8]
            }

gbm = xgb.XGBClassifier(
                       n_estimators=30000, #arbitrary large number
                       max_depth=5, # 3
                       objective='binary:logistic',
                       learning_rate=.1,
                       subsample=1,
                       min_child_weight=1,
                       colsample_bytree=.8
                      )

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=30, shuffle=True)
eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)] #tracking train/validation error as we go

fit_model = gbm.fit(
                    X_train_sub, y_train_sub,
                    eval_set=eval_set,
                    eval_metric='auc',
                    early_stopping_rounds=20,
                    verbose=False #gives output log as below
                   )

get_metrics_and_roc_curve(gbm, 'xgb', X_test, y_test)
#================================================================================================================================



#==============================================================================================================================
# LOGISTIC REGRESSION MODEL

for_modeling = pd.read_csv(data_directory+'clean_transformed_data_for_simple_model2.csv')
# for_modeling = pd.read_csv(data_directory+'subset_model_w_listeing.csv')

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

#==============================================================================================================================
# CORRELATION OF VARIABLES

# city and gender 0.87
# city and registration 0.63
# city and bd 0.87
# gender and registered_via 0.59
# gender and bd 0.95
# registered_via and bd
# reg_month and reg_day
for_corr = overlap.copy()
for_corr.append('is_churn_final')
overlap
cor = for_modeling[for_corr]
cor = cor.corr()
cor
# cor[cor.values > 0.5]

#==============================================================================================================================
# OLD

pred_proba = log_reg.predict_proba(X_test)[:,1]
proba_inverseT = sc.inverse_transform(X_test)
proba_df = pd.DataFrame(proba_inverseT)
proba_df['pred_proba'] = pred_proba
proba_df.columns = ['city_woe', 'registered_via_woe', 'bd', 'pred_proba']
proba_df = proba_df.sort_values(by='pred_proba', ascending=False)
proba_df[(proba_df.bd != 999) & (proba_df.pred_proba < 0.65) & (proba_df.pred_proba > 0.55)]
# city_woe = 0.355208, registered_via_woe = -1.017460, age = 15, pred_proba = 0.795147
# city_woe = -0.406446, registered_via_woe = -0.209556, age = 30, pred_proba = 0.555077
# city_woe = -0.522860, registered_via_woe = 0.820627, age = 49, pred_proba = 0.300634
