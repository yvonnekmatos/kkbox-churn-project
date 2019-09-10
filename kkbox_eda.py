import pandas as pd
import numpy as np
from wd import *

# We can predict if customers will cancel a subscription (normal classification problem)
# For customers that do cancel, can we predict when they will cancel? (May not be enough data)

def get_final_target(churn, update):
    if np.isnan(update):
        return churn
    elif np.isnan(churn):
        return update
    else:
        return update

# COMBINE THE TARGET VARIABLES
# One observation = unique customer ID and whether they churned
# There are 1,082,190 unique customers

target = pd.read_csv(data_directory+'train.csv')
len(target)
target.head()
target.is_churn.mean()
len(target.groupby('msno').size())

target2 = pd.read_csv(data_directory+'train_v2.csv')
target2.columns = ['msno', 'is_churn_update']
len(target2)
len(target2.groupby('msno').size())


total_target = pd.merge(target, target2, on='msno', how='outer')
len(total_target)
total_target['is_churn_final'] = total_target.apply(lambda row: get_final_target(row.is_churn, row.is_churn_update), axis=1)
# 12.4% churn
total_target.is_churn_final.mean()
total_target.info()

# 134,480 customers canceled
total_target.is_churn_final.value_counts()

# total_target.to_csv(data_directory+'final_churn_target.csv', index=False)
#==================================================================================================================================
# MEMBERS DATA

members = pd.read_csv(data_directory+'members_v3.csv', nrows=1000)
len(members)
members.head()
members.msno.nunique()

members.describe()

mem_target = pd.merge(total_target, members, on='msno', how='inner')
# member info for 88% of customers in total_target
len(mem_target)/len(total_target)
mem_target.groupby('registered_via').is_churn_final.mean()

# TRY VERY SIMPLE MODEL WITH MEMBERS DATA just out of curiousity
# I expect behavioral data will be very important for predicting churn
# City to str for categorical - run WOE
# Clean up bd (age) to make sense
# Gender to categorical and nulls to 'unknown' - run WOE

# Membership length


#==================================================================================================================================
# TRANSACTIONS DATA
# (This may be the trickiest part)
# One observation = one transaction per customer per date (can be multiple transactions on a single date)

transactions = pd.read_csv(data_directory+'transactions.csv')
len(transactions)
transactions.msno.nunique()
transactions = transactions.sort_values(by='msno')
transactions.tail(10)
transactions.info()

transactions2 = pd.read_csv(data_directory+'transactions_v2.csv')
len(transactions2)
transactions2.msno.nunique()

all_transactions = transactions.append(transactions2)
# all_transactions.to_csv(data_directory+'all_transactions.csv', index=False)
all_transactions.head()
# No nulls in all_transactions
all_transactions.info()
all_transactions.describe()

all_transactions['outstanding_bal'] = all_transactions.plan_list_price - all_transactions.actual_amount_paid

total_target = pd.read_csv(data_directory+'final_churn_target.csv')
all_trans_target = pd.merge(total_target[['msno', 'is_churn_final']], all_transactions, on='msno', how='left')
len(all_trans_target.groupby('msno').size())

all_trans_target['transaction_date'] = pd.to_datetime(all_trans_target.transaction_date.astype(str), format='%Y%m%d')
all_trans_target['membership_expire_date'] = pd.to_datetime(all_trans_target.membership_expire_date.astype(str), format='%Y%m%d')
all_trans_target.head()
all_trans_target.to_csv(data_directory+'all_transactions_in_w_target_dt.csv', index=False)


# Check to see if there are no rows in transactions2 that are an update of rows in transactions
trans_per_date = transactions.groupby(['msno', 'transaction_date', 'membership_expire_date']).size()
trans_per_date2 = transactions2.groupby(['msno', 'transaction_date', 'membership_expire_date']).size()

(len(trans_per_date) + len(trans_per_date2)) == len(all_transactions.groupby(['msno', 'transaction_date', 'membership_expire_date']).size())
# all_tranasactinos is good! yay


# How many users have more than one transaction per transaction_date and membership_expire_date?
trans_per_date_df = pd.DataFrame(trans_per_date).reset_index()
len(trans_per_date_df[trans_per_date_df[0]!=1])
trans_per_date_df[0].max()
trans_per_date_df[trans_per_date_df[0]==9]
many_trans_daily = list(trans_per_date_df[trans_per_date_df[0]==9]['msno'].values)

mask = (all_transactions.msno == many_trans_daily[0]) & \
        (all_transactions.transaction_date == 20161017) & \
        (all_transactions.membership_expire_date == 20161116) # 20161017
all_transactions[mask]
all_transactions.head()
all_transactions.msno.nunique()
len(all_transactions.groupby(['msno', 'plan_list_price']).size())

# FEATURES TO ENGINEER FROM THIS DF
# Simple features on large df
# DONE plan_list_price - actual_amount_paid
# Number of payment_method_id's used per msno
# number of payment plans per user?
pmt_ids_per_user = pd.DataFrame(all_transactions.groupby(['msno', 'payment_method_id']).size()).reset_index()
# merge this back in to all_transactions
pmt_ids_per_user

# Try these pandas functions with dask??

# More complex features
# Number of times (transactions per date > 1) - group by plan_list_price
# Number of plans at once?
#==================================================================================================================================
all_trans_target = pd.read_csv(data_directory+'all_transactions_in_w_target.csv')
all_trans_target['transaction_date'] = pd.to_datetime(all_trans_target.transaction_date.astype(str), format='%Y%m%d')
all_trans_target['membership_expire_date'] = pd.to_datetime(all_trans_target.membership_expire_date.astype(str), format='%Y%m%d')
all_trans_target.transaction_date[0]
all_trans_target.head()
mem_target = pd.read_csv(data_directory+'clean_members_w_target.csv', parse_dates=['registration_init_time'])
mem_target = mem_target[mem_target.reg_year >= 2015]
mem_target.info()

all_trans_target.info()
mem_target['transaction_date'] = mem_target.registration_init_time
mem_target.msno[0]

trans_mem_target = pd.merge(all_trans_target, mem_target[['msno', 'transaction_date']], \
                    on=['msno', 'transaction_date'], how='outer')
trans_mem_target[trans_mem_target.msno == mem_target.msno[0]]
all_trans_target.transaction_date.min()

trans_mem_target.transaction_date.min()


trans_mem_target.to_csv(data_directory+'transactions_w_reg_init_date_after_2015.csv', index=False)
#==================================================================================================================================
# USER LOGS DATA
# One observation = aggregated stats about songs played per user per date

# huge data files - logs has 392,106,543 (must use PySpark, too big for Pandas to handle)
# can I just append logs2 to logs?

logs = pd.read_csv(data_directory+'user_logs.csv', nrows=100000)
len(logs)
logs.head(10)

logs.info()
logs_churn = pd.merge(logs, total_target[['msno', 'is_churn_final']], on='msno', how='left')
logs_churn.iloc[2:15,:]

logs2 = pd.read_csv(data_directory+'user_logs_v2.csv')
len(logs2)
logs2.columns = logs2.columns.map(lambda col: col+'_update' if col != 'msno' and col != 'date' else col)
logs2.head()

all_logs = pd.merge(logs, logs2, on=['msno', 'date'], how='outer')
