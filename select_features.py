import pandas as pd
import numpy as np
from wd import *
from helpful_functions import *
import seaborn as sns
from datetime import datetime
import datetime as dt
from os import listdir
from itertools import combinations
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=Warning)


# GET FEATURE IMPORTANCE FOR ALL FEATURES INCLUDING USER LOGS ENGINEERED FEATURES
# REMOVE COLLINEAR VARIABLES 

def read_files_compile_df(path):
    file_list = [file for file in listdir(path) if file not in ['_SUCCESS', '.DS_Store']]
    total_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(path+'/'+file)
        total_df = total_df.append(df)
    return total_df

logs_2nd_last_row_per_user = read_files_compile_df(data_directory+'logs_2nd_last_row_per_user')
logs_2nd_last_row_per_user.shape

logs_3rd_last_row_per_user = read_files_compile_df(data_directory+'logs_3rd_last_row_per_user')
logs_3rd_last_row_per_user.shape

logs_last_row_per_user = read_files_compile_df(data_directory+'logs_last_row_per_user')
logs_last_row_per_user.shape

logs_sum_last_3_rows_per_user = read_files_compile_df(data_directory+'logs_sum_last_3_rows_per_user')
logs_sum_last_3_rows_per_user.shape

logs_sum_of_rows_per_user = read_files_compile_df(data_directory+'logs_sum_of_rows_per_user')
logs_sum_of_rows_per_user.shape

for_modeling = pd.read_csv(data_directory+'clean_transformed_data_for_simple_model2.csv')
for_modeling.shape
for_modeling.columns

all_features = pd.merge(for_modeling, logs_2nd_last_row_per_user, on='msno', how='left')
all_features.shape

df_list = [logs_3rd_last_row_per_user, logs_last_row_per_user, logs_sum_last_3_rows_per_user, logs_sum_of_rows_per_user]

for df in df_list:
    all_features = pd.merge(all_features, df, on='msno', how='left')

all_features.shape

all_features.info()

all_features_no_dates = all_features.drop(['date_2ndlast', 'date_3rdlast', 'date_last', 'row_count_3last', 'row_count'], axis=1)

all_features_no_dates.describe().T


for_corr = all_features_no_dates.drop(['msno'
                              , 'sum_abs_total_secs'
                              , 'sum_abs_total_secs_3last'
                              , 'abs_total_secs_last'
                              , 'abs_total_secs_3rdlast'
                              , 'abs_total_secs_2ndlast'], axis=1)

for_corr.describe().T

# Fill in missing values
corr_filled = for_corr.fillna(-5)
corr_filled.info()
#========================================================================================================================================
# Recursive feature elimination to get rank order of feature importance

X = corr_filled.drop('is_churn_final', axis=1)
y = corr_filled.is_churn_final

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=19, shuffle=True)

X_train, y_train = under_sample_data(X_train,y_train)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=1)
selector = selector.fit(X_train, y_train)

feature_rank = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), list(X)))
feature_rank

#========================================================================================================================================
# DROP COLLINEAR FEATURES

# Get list of unique combinations of variables
all_cols = list(corr_filled)
features = deepcopy(all_cols)
features.remove('is_churn_final')

col_combinations = [list(comb) for comb in combinations(features, 2)]
len(col_combinations)

# Get feature pair combinations that have a correlation > 0.5
collinear_vars = []
for comb in col_combinations:
    corr = corr_filled[comb].corr()
    if abs(corr.iloc[1,0]) > 0.5:
        collinear_vars.append(comb)

# Feature names in ranked order
all_feat = [elem[1] for elem in feature_rank]


def eliminate_collinear_features(all_feat, collinear_vars):
    big_ls = []
    # Taking pairs in feature rank (1,2) (2,3) ...
    for first, second in zip(all_feat, all_feat[1:]):
        ls = []
        ls.append(first)
        ls.append(second)
        big_ls.append(ls)

    coll_pairs = []
    for elem in big_ls:
        if elem in collinear_vars or elem[::-1] in collinear_vars:
            coll_pairs.append(elem)

    exclude = [elem[1] for elem in coll_pairs]
    round_1_keep = [elem for elem in all_feat if elem not in exclude]
    return round_1_keep

# Call until rounds match
round_1_keep = eliminate_collinear_features(all_feat, collinear_vars)
round_2_keep = eliminate_collinear_features(round_1_keep, collinear_vars)
round_1_keep == round_2_keep

len(round_1_keep)
round_1_keep

corr_filled.info()

model = corr_filled[round_1_keep]
model.shape
model = model.join(for_modeling[['msno', 'is_churn_final']])
model.shape
list(model)

model.to_csv(data_directory+'rfe_reduced_feature_space.csv', index=False)
