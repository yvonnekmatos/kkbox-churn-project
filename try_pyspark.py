import pyspark
from wd import *
from pyspark.sql.functions import datediff, to_date, lit
from pyspark.sql import functions as F
# from pyspark import SparkContext
# from pyspark import SQLContext
from pyspark.sql.window import Window



spark = pyspark.sql.SparkSession.builder.getOrCreate()

# All transactions in churn dataset, without registration init date
# transactions = spark.read.csv(data_directory+'all_transactions_in_w_target.csv', header=True, inferSchema=True)

# All transactions in churn dataset, including registration init date, with registration init date 2015 or later
transactions = spark.read.csv(data_directory+'transactions_w_reg_init_date_after_2015.csv', header=True, inferSchema=True)

# All transactions in churn dataset, including registration init date
# transactions = spark.read.csv(data_directory+'all_transactions_w_reg_init_date.csv', header=True, inferSchema=True)
transactions.show(5)
transactions.printSchema()
# transactions = transactions.withColumn('transaction_date', transactions.transaction_date.cast('string'))
# transactions = transactions.withColumn('transaction_date', F.to_date(F.unix_timestamp('transaction_date', 'yyyyMMdd').cast('timestamp')))
# transactions = transactions.join(mem_target.select(['mem_msno', 'transaction_date']), \
#                                 (mem_target.mem_msno == transactions.msno) & (mem_target.transaction_date == transactions.transaction_date), how='left')

my_window = Window.partitionBy('msno').orderBy('transaction_date')

transactions = transactions.withColumn('prev_transaction_date', F.lag(transactions.transaction_date).over(my_window))

# transactions = transactions.withColumn('days_since_prev_transaction', F.when(F.isnull(df.value - df.prev_value), 0)
#                               .otherwise(df.value - df.prev_value))
transactions.show(50)

# transactions = transactions.withColumn('transaction_date', transactions.transaction_date.cast('string'))
# transactions = transactions.withColumn('transaction_date', F.to_date(F.unix_timestamp('transaction_date', 'yyyyMMdd').cast('timestamp')))
# transactions = transactions.withColumn('prev_transaction_date', transactions.prev_transaction_date.cast('string'))
# transactions = transactions.withColumn('prev_transaction_date', F.to_date(F.unix_timestamp('prev_transaction_date', 'yyyyMMdd').cast('timestamp')))

transactions = transactions.withColumn('days_since_prev_transaction',\
    F.datediff(transactions.transaction_date, transactions.prev_transaction_date))

transactions = transactions.withColumn('months_since_prev_transaction',\
    F.months_between(transactions.transaction_date, transactions.prev_transaction_date))
transactions.printSchema()
transactions.select(['msno', 'is_churn_final', 'payment_plan_days', 'transaction_date', 'prev_transaction_date', 'months_since_prev_transaction'])\
.filter(transactions.months_since_prev_transaction >= 2).groupby('msno').count().count()
# Insert extra rows where months_since_prev_transaction > 1
# Label row numbers (month) through partition query

# MVP filter out all users that have months_since_prev_transaction < 2
transactions.groupby('msno').count().count()

transactions.show(50)

transactions.groupby('msno').filter(transactions.transaction_date > F.to_date(F.unix_timestamp('20150101', 'yyyyMMdd')))

transactions.createOrReplaceTempView('trans')
# filter on transactions where min transaction date >= 2015-01-01
# 'SELECT msno, is_churn_final, transaction_date, prev_transaction_date, months_since_prev_transaction ' \
q = 'SELECT * ' \
    'FROM trans ' \
    'GROUP BY msno ' \
    'HAVING min(transaction_date) >= 2015-01-01'
joined_2015_or_later = spark.sql(q)


#==============================================================================================================================

transactions.groupby('msno').count().count()
transactions.schema.names

transactions.createOrReplaceTempView('trans')
# Getting last row of transactions data per customer
inner_query = 'with row_per_group as (' \
                    'SELECT * ' \
                    ', row_number() over (PARTITION BY msno ORDER BY transaction_date DESC) as row_number ' \
                    'from trans) ' \
                'SELECT *, plan_list_price - actual_amount_paid as outstanding_bal ' \
                'FROM row_per_group ' \
                'WHERE row_number = 1'
last_transaction_per_user = spark.sql(inner_query)
last_transaction_per_user.show(10)


# Getting Id's that one transaction per date
q = 'SELECT msno ' \
    'FROM trans ' \
    'GROUP BY msno, transaction_date, membership_expire_date ' \
    'HAVING count(payment_method_id) = 1'
one_transaction_per_date = spark.sql(q)
one_transaction_per_date.show(10)



# FEATURES I WANT
# Last row per msno and transaction_date
# Count of num payment_method_id, payment_plan_days, plan_list_price,
# Sum of is_cancel

# EXAMPLE CODE FOR LATER
# target = target.withColumn('msno_target', target.msno).select(['msno_target','msno'])
# trans_target = target.select(['msno_target']).join(transactions, target.msno_target == transactions.msno,how='left')
# trans_target.show(5)
# trans_target.schema.names

# mem_target = spark.read.csv(data_directory+'clean_members_w_target.csv', header=True, inferSchema=True)
# mem_target.printSchema()
# mem_target = mem_target.withColumn('transaction_date', mem_target.registration_init_time)
# mem_target = mem_target.withColumn('mem_msno', mem_target.msno)
# mem_target.show(5)


# For Survival Analysis:
# ID (index), Month since registration_init_time (1,2,3), Treatment (I don't think I need this?, not compraing products),
# Event (0 or 1 is_churn_final)
# Will have right-censored data (customers that don't churn) - must of course keep in model

# Normal survival analysis looks at time to churn without other features
# Hazard function = probability of churn occurring at a given time (at the population level?)
# Cox proportional hazard (survival regressor) = probability of churn occurring at a given time (at the population level?)

# Could do survival analysis just on customers that stay with you the whole time (autorenew or not but re-enroll every month)
# for customers that have a subscription gap (autorenew/don't and have a month here and there)
# customers can also autorenew for an extended amount of time
# PySpark has AFTSurvivalRegression implemented!
#===========================================================================================================
# USER LOGS DATA
logs = spark.read.csv(data_directory+'user_logs.csv', header=True, inferSchema=True)
# logs.count()
# logs.printSchema()
#
# logs.show(5)

logs2 = spark.read.csv(data_directory+'user_logs_v2.csv', header=True, inferSchema=True)
# logs2.show(5)
# logs2.count()

all_logs = logs.union(logs2)
df.write.csv('mycsv.csv')
# all_logs.show(5)
# all_logs.count()
# 392106543+18396362

# Everything appended correctly
# No overlap when grouping by msno and date
# all_logs.select(['msno', 'date']).groupby(['msno', 'date']).mean().count()
#
# all_logs.describe().show()

# all_logs = all_logs.orderBy(['msno', 'date'])
logs2 = logs2.limit(1000)
# all_logs_sub = all_logs_sub.orderBy(['msno', 'date'])
logs2.createOrReplaceTempView('all_logs')


spark.sql('select * from all_logs').show(5)

'group by msno)' \
inner_query = 'with row_per_group as (' \
                    'SELECT * ' \
                    ', row_number() over (partition by msno ORDER BY date) as row_number ' \
                    'from all_logs) ' \
                'SELECT * ' \
                'FROM row_per_group ' \
                'WHERE row_number = 2'
q2 = spark.sql(inner_query)
q2.show()


part_query = 'SELECT row_number() over (partition by msno ORDER BY date) as row_number ' \
                'from all_logs'
spark.sql(part_query).show(10)

last_row_per_user_query = 'SELECT al.msno ' \
                    ', al.date ' \
                    ', num_25 as num_25_last ' \
                    ', num_50 as num_50_last ' \
                    ', num_75 as num_75_last ' \
                    ', num_985 as num_985_last ' \
                    ', num_100 as num_100_last ' \
                    ', num_unq as num_unq_last ' \
                    ', total_secs as total_secs_last ' \
              'from ' \
                    '(SELECT msno, max(date) as date FROM all_logs GROUP BY msno) md ' \
              'left join all_logs al ' \
              'on al.msno = md.msno AND al.date = md.date'
q2 = spark.sql(last_row_per_user_query)
q2.show()



query = 'SELECT msno ' \
            ', sum(num_25) as sum_num_25 ' \
            ', sum(num_50) as sum_num_50 ' \
            ', sum(num_75) as sum_num_75 ' \
            ', sum(num_985) as sum_num_985 ' \
            ', sum(num_100) as sum_num_100 ' \
            ', sum(num_unq) as sum_num_unq ' \
            ', sum(case when total_secs < 0 then 0 else total_secs end) as sum_total_secs_clean ' \
            ', sum(abs(total_secs)) as sum_abs_total_secs ' \
         'FROM all_logs ' \
         'group by msno'

q = spark.sql(query)

q.show(5)
# FEATURES TO CREATE FROM LOGS TABLE
# Min date col
# max date col
# num days between max and min dates (will do this again when merging with members/transactions on registration_init_time)
# last 2-3 entries (as two sets of cols)
# DONE sum of each col (except date) for each msno

#===========================================================================================================
# PY CODE
logs = pd.read_csv(data_directory+'user_logs.csv')

logs2 = pd.read_csv(data_directory+'user_logs_v2.csv')
len(logs2)
logs2.columns = logs2.columns.map(lambda col: col+'_update' if col != 'msno' and col != 'date' else col)
logs2.head()

all_logs = pd.merge(logs, logs2, on=['msno', 'date'], how='outer')
