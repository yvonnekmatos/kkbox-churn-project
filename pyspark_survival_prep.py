import pyspark
from wd import *
from pyspark.sql.functions import datediff, to_date, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
import pandas as pd

spark = pyspark.sql.SparkSession.builder.getOrCreate()

transactions = spark.read.csv(data_directory+'transactions_w_reg_init_date_after_2015.csv', header=True, inferSchema=True)
transactions.count()
transactions.printSchema()
transactions = transactions.select(['msno', 'is_churn_final', 'transaction_date'])

my_window = Window.partitionBy('msno').orderBy('transaction_date')

transactions = transactions.withColumn('prev_transaction_date', F.lag(transactions.transaction_date).over(my_window))

transactions = transactions.withColumn('months_since_prev_transaction',\
    F.months_between(transactions.transaction_date, transactions.prev_transaction_date))
transactions.printSchema()
transactions = transactions.where(F.col('months_since_prev_transaction').isNotNull())




transactions.createOrReplaceTempView('trans')
q = 'SELECT first(msno) as msno, first(is_churn_final) as is_churn_final, round(sum(months_since_prev_transaction), 0) as month ' \
    'FROM trans ' \
    'GROUP BY msno '
joined_2015_or_later = spark.sql(q)

joined_2015_or_later.show(50)

joined_2015_or_later = joined_2015_or_later.toPandas()
joined_2015_or_later.head()
joined_2015_or_later.is_churn_final.mean()
joined_2015_or_later.shape

joined_2015_or_later.to_csv(data_directory+'survival_data/survival_2015_after.csv', index=False)



#========================================================================================================================================


# exclude = transactions.select(['msno', 'is_churn_final', 'transaction_date', 'prev_transaction_date', 'months_since_prev_transaction'])\
#     .filter(transactions.months_since_prev_transaction >= 1.5).groupby('msno').count()
#
#
# exclude_array = [row.msno for row in exclude.select('msno').collect()]
#
# transactions = transactions.filter(~transactions.msno.isin(exclude_array))
# transactions.\
# select(['msno', 'is_churn_final', 'payment_plan_days', 'transaction_date', 'prev_transaction_date', 'months_since_prev_transaction']).show(50)

def ceil_or_floor(x):
    if x < 0.5:
        return float(0)
    elif 0.5 <= x < 1:
        return float(1)
    elif x > 1:
        return float(1)
    else:
        return float(x)

udf_round = udf(lambda x: ceil_or_floor(x))


transactions = transactions.withColumn('round_months_since_prev_transaction', udf_round(transactions.months_since_prev_transaction))
transactions.show(50)

transactions = transactions.select(['msno', 'is_churn_final', 'transaction_date', \
            'prev_transaction_date', 'months_since_prev_transaction', 'round_months_since_prev_transaction'])\
            .where(F.col('round_months_since_prev_transaction') == 1)


transactions.count()

my_window_desc = Window.partitionBy('msno').orderBy(F.desc('transaction_date'))

transactions = transactions.withColumn('month_desc', F.row_number().over(my_window_desc))
transactions = transactions.withColumn('month', F.row_number().over(my_window))

transactions = transactions.where(F.col('month_desc') == 1)

transactions.show(50)
transactions.count()

pd_transactions = transactions.toPandas()
pd_transactions.head()
pd_transactions.to_csv(data_directory+'survival_data/survival_subset.csv', index=False)
