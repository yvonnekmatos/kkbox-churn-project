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
mem_target.to_csv(data_directory+'clean_members_w_target.csv', index=False)
#==================================================================================================================================
# CLEANED DATASET READY FOR SIMPLE MODEL

mem_target = pd.read_csv(data_directory+'clean_members_w_target.csv')
mem_target['city'] = mem_target.city.astype(str).fillna('999')
mem_target['registered_via'] = mem_target.registered_via.astype(str).fillna('999')
mem_target = mem_target.drop('registration_init_time', axis=1)

mem_target.info()
for_woe = mem_target.drop(['bd', 'reg_year'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(for_woe, for_woe.is_churn_final, test_size=0.4, random_state=19, shuffle=True)
X_train.is_churn_final.mean()
X_test.is_churn_final.mean()


features_id = list(X_train)
features = features_id[2:]
features

iv_dict, woe_df, X_train_cat_feat_woe_transformed = calc_woe_and_transform_many_features(X_train, features, features_id, 'is_churn_final')

X_test_cat_feat_woe_transformed = woe_transform_many_features(X_train, woe_df, features, features_id, 'is_churn_final')

total_cat_feat_woe_transformed = X_train_cat_feat_woe_transformed.append(X_test_cat_feat_woe_transformed)

total_cat_feat_woe_transformed = total_cat_feat_woe_transformed.drop(features, axis=1)

total_cat_feat_woe_transformed.info()

mem_target.columns

for_modeling = pd.merge(total_cat_feat_woe_transformed, mem_target[['msno', 'bd', 'reg_year']], on='msno', how='inner')

for_modeling.info()


X = for_modeling.drop(['msno', 'is_churn_final'], axis=1)
y = for_modeling.is_churn_final

for_modeling.to_csv(data_directory+'clean_transformed_data_for_simple_model.csv', index=False)


mem_target.columns
mem_target.groupby(['reg_year']).is_churn_final.mean()
mem_target.groupby(['reg_year']).agg({'is_churn_final': ['size', 'mean']}).reset_index()


# Over weekend, get another minimum feature of latest month's listening behavior, and latest month's transactions per account to try in baseline model
# Try this using dask.
# side note for evening: XGB parameter tuning using gridsearchCV

# PUT OFF Try featuretools with a subset of data using Python to get familiar with it.
# Next week: Survival analysis to find when customers will typically leave

# MAYBE: Next week: work on scaling up feature engineering using featuretools and pyspark in the cloud

# Business questions:
# Which customers will leave? (churn model)
# When do customers typically leave us? (survival analysis with hazard function)
# Which customers will make us the most money? (clustering)

# Survival analysis: summarize how many customers leave at each time point (and what they look like in profile)
# Clustering: help get customer profiles to identify high money makers and what they look like, least money and what they look like

# Questions:
# 1. Plan going forward sound good?
# 2. What else can I do with this dataset?
# 3. XGB did not outperform RF - normal?

# Time allotment
# 1 d to set up cluster for PySpark, EMR, 1d for pyspark code, try to request larger computer for EMR, EC2 - get that approval now, it can take a few days


# TRY VERY SIMPLE MODEL WITH MEMBERS DATA just out of curiousity
# I expect behavioral data will be very important for predicting churn
# City to str for categorical - run WOE
# Clean up bd (age) to make sense
# Gender to categorical and nulls to 'unknown' - run WOE
#
