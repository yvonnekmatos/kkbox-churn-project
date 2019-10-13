import pyspark
from wd import *
from pyspark.sql.functions import datediff, to_date, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
import pandas as pd

# GET MEMBERSHIP LENGTH FOR CUSTOMERS WHO JOINED IN 2015 OR LATER 

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
