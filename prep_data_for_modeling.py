import pandas as pd
import numpy as np
from wd import *
from helpful_functions import *
import seaborn as sns
from datetime import datetime
import datetime as dt
from sklearn.model_selection import train_test_split, KFold, cross_val_score

import warnings
warnings.simplefilter(action='ignore', category=Warning)

#==================================================================================================================================
# CUSTOM FUNCTIONS

def fix_bd(x):
    if 12 <= x <= 85:
        return x
    else:
        return 999

def calc_woe_and_transform_one_feature(main, feature, target):
    iv, woe = calc_iv(main, feature, target)
    woe_val = clean_columns(woe)
    woe_val = woe_val.rename(columns={'value': feature, 'woe': feature+'_woe'})
    woe_val = woe_val.iloc[:,[1,-2]]
    return iv, woe, woe_val

def calc_woe_and_transform_many_features(main, features, features_id, target):
    main_categoricals_woe_transformed = main.copy()
    main_categoricals_woe_transformed = main_categoricals_woe_transformed[features_id]
    main_categoricals_woe_transformed = main_categoricals_woe_transformed.fillna('NULL')

    woe_df = pd.DataFrame()
    iv_dict = {}
    for feature in features:
        iv, woe, woe_val = calc_woe_and_transform_one_feature(main, feature, target)
        iv_dict[feature] = iv
        woe_df = woe_df.append(woe)
        main_categoricals_woe_transformed = pd.merge(main_categoricals_woe_transformed, woe_val, on=feature, how='left')
    return iv_dict, woe_df, main_categoricals_woe_transformed

def woe_transform_one_feature(main, woe_df, feature, target):
    woe_val = clean_columns(woe_df)
    woe_val = woe_df[woe_df.variable==feature]
    woe_val = woe_val.rename(columns={'value': feature, 'woe': feature+'_woe'})
    woe_val = woe_val.iloc[:,[1,-2]]
    return woe_val

def woe_transform_many_features(main, woe_df, features, features_id, target):
    main_categoricals_woe_transformed = main.copy()
    main_categoricals_woe_transformed = main_categoricals_woe_transformed[features_id]
    main_categoricals_woe_transformed = main_categoricals_woe_transformed.fillna('NULL')

    for feature in features:
        woe_val = woe_transform_one_feature(main, woe_df, feature, target)
        main_categoricals_woe_transformed = pd.merge(main_categoricals_woe_transformed, woe_val, on=feature, how='left')
    return main_categoricals_woe_transformed


#==================================================================================================================================
# TARGET DATA

total_target = pd.read_csv(data_directory+'final_churn_target.csv')
total_target = total_target[['msno', 'is_churn_final']]
len(total_target)
#==================================================================================================================================
# MEMBERS DATA

members = pd.read_csv(data_directory+'members_v3.csv')
len(members)
members.head()
members.msno.nunique()

members.describe()

# member info for 88% of customers in total_target
mem_target = pd.merge(total_target, members, on='msno', how='left')
len(mem_target)
mem_target.head()
mem_target.groupby('registered_via').is_churn_final.mean()
mem_target.info()
mem_target['city'] = mem_target.city.astype(str).fillna('999')
mem_target['registered_via'] = mem_target.registered_via.astype(str).fillna('999')
mem_target['gender'] = mem_target.gender.fillna('unknown')
mem_target['bd'] = mem_target.bd.fillna(999).apply(fix_bd)

mem_target['registration_init_time'] = pd.to_datetime(mem_target.registration_init_time.astype(str), format='%Y%m%d')
mem_target['reg_year'] = mem_target.registration_init_time.dt.year.fillna(999)
mem_target['reg_month'] = mem_target.registration_init_time.dt.month_name().fillna('999')
mem_target['reg_day_of_week'] = mem_target.registration_init_time.dt.day_name().fillna('999')

mem_target.head()

mem_target.reg_day_of_week.value_counts()
mem_target.info()
mem_target.msno.nunique()
mem_target.to_csv(data_directory+'clean_members_w_target.csv', index=False)
#==================================================================================================================================
# CLEANED DATASET READY FOR SIMPLE MODEL

mem_target = pd.read_csv(data_directory+'clean_members_w_target.csv')
mem_target.msno.nunique()

mem_target['city'] = mem_target.city.astype(str).fillna('999')
mem_target['registered_via'] = mem_target.registered_via.astype(str).fillna('999')
mem_target = mem_target.drop('registration_init_time', axis=1)

mem_target.info()
for_woe = mem_target.drop(['bd', 'reg_year'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(for_woe, for_woe.is_churn_final, test_size=0.4, random_state=19, shuffle=True)
X_train.is_churn_final.mean()
X_test.is_churn_final.mean()

X_train.shape
X_train.msno.nunique()
X_test.shape
X_test.msno.nunique()

len(X_train)+len(X_test)

features_id = list(X_train)
features = features_id[2:]
features

iv_dict, woe_df, X_train_cat_feat_woe_transformed = calc_woe_and_transform_many_features(X_train, features, features_id, 'is_churn_final')
X_train_cat_feat_woe_transformed.shape[0] == X_train.shape[0]
X_train_cat_feat_woe_transformed.msno.nunique()

# Need to debug here
X_test_cat_feat_woe_transformed = woe_transform_many_features(X_test, woe_df, features, features_id, 'is_churn_final')
X_test_cat_feat_woe_transformed.shape[0] == X_test.shape[0]
X_test_cat_feat_woe_transformed.shape[0]
X_test.shape[0]

X_test_cat_feat_woe_transformed


total_cat_feat_woe_transformed = X_train_cat_feat_woe_transformed.append(X_test_cat_feat_woe_transformed)
total_cat_feat_woe_transformed.shape
total_cat_feat_woe_transformed.msno.nunique()

total_cat_feat_woe_transformed = total_cat_feat_woe_transformed.drop(features, axis=1)

total_cat_feat_woe_transformed.info()

mem_target.columns

for_modeling = pd.merge(total_cat_feat_woe_transformed, mem_target[['msno', 'bd', 'reg_year']], on='msno', how='inner')

for_modeling.info()

for_modeling.msno.nunique()


for_modeling.to_csv(data_directory+'clean_transformed_data_for_simple_model2.csv', index=False)