import pyspark
from wd import *
from pyspark.sql.functions import datediff, to_date, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
# import pandas as pd

# INSPECT AND CLEAN USER LOGS DATA IN PREPARATION FOR CREATING SCRIPT TO RUN ON EMR

spark = pyspark.sql.SparkSession.builder.getOrCreate()

# USER LOGS DATA
logs = spark.read.csv(data_directory+'user_logs.csv', header=True, inferSchema=True)

logs.count()
# No duplicate date entries per user
logs.groupby(['msno', 'date']).count().count()
# Negative values where there shouldn't be? Total secs has values < 0
logs.describe().show()


logs2 = spark.read.csv(data_directory+'user_logs_v2.csv', header=True, inferSchema=True)


logs2.show(5)
logs2.count()
# No duplicate date entries per user
logs2.groupby(['msno', 'date']).count().count()
# No negative values where there shouldn't be
logs2.describe().show()

all_logs = logs.union(logs2)
all_logs.show(5)
all_logs.count()
logs.count() + logs2.count()
# Everything appended correctly

# No overlap when grouping by msno and date
all_logs.groupby(['msno', 'date']).mean().count()



# all_logs.createOrReplaceTempView('all_logs')
logs2.createOrReplaceTempView('all_logs')
# logs.createOrReplaceTempView('all_logs')

logs2.printSchema()
# GETTING ROWS PER USER
last_row_per_user_query = 'WITH row_per_user as ( \
                    SELECT *  \
                    , row_number() OVER (PARTITION BY msno ORDER BY date DESC) as row_number  \
                    FROM all_logs)  \
                SELECT msno  \
                    , date as date_last \
                    , num_25 as num_25_last \
                    , num_50 as num_50_last \
                    , num_75 as num_75_last \
                    , num_985 as num_985_last \
                    , num_100 as num_100_last \
                    , num_unq as num_unq_last \
                    , CASE WHEN total_secs < 0 THEN 0 ELSE total_secs END as total_secs_clean_last  \
                    , ABS(total_secs) as abs_total_secs_last  \
                FROM row_per_user  \
                WHERE row_number = 1'
last_row_per_user = spark.sql(last_row_per_user_query)
last_row_per_user.write.save(data_directory+'TEST_logs_last_row_per_user.csv', format='csv', header=True)
row_per_user.count()
row_per_user.show()
row_per_user_pd = row_per_user.toPandas()
row_per_user_pd.to_csv(data_directory+'logs_last_row_per_user.csv', index=False)

last_row_per_user.toPandas().to_csv(data_directory+'test.csv', index=False)

# 2nd to last row per user
second_last_row_per_user_query = 'WITH row_per_user as ( \
                    SELECT *  \
                    , row_number() OVER (PARTITION BY msno ORDER BY date DESC) as row_number  \
                    FROM all_logs)  \
                SELECT msno  \
                    , date as date_2ndlast \
                    , num_25 as num_25_2ndlast \
                    , num_50 as num_50_2ndlast \
                    , num_75 as num_75_2ndlast \
                    , num_985 as num_985_2ndlast \
                    , num_100 as num_100_2ndlast \
                    , num_unq as num_unq_2ndlast \
                    , CASE WHEN total_secs < 0 THEN 0 ELSE total_secs END as total_secs_clean_2ndlast  \
                    , ABS(total_secs) as abs_total_secs_2ndlast  \
                FROM row_per_user  \
                WHERE row_number = 2'
second_last_row_per_user = spark.sql(second_last_row_per_user_query)
second_last_row_per_user.toPandas().to_csv(data_directory+'test.csv', index=False)

# 3rd to last row per user
third_last_row_per_user_query = 'WITH row_per_user as ( \
                    SELECT *  \
                    , row_number() OVER (PARTITION BY msno ORDER BY date DESC) as row_number  \
                    FROM all_logs)  \
                SELECT msno  \
                    , date as date_3rdlast \
                    , num_25 as num_25_3rdlast \
                    , num_50 as num_50_3rdlast \
                    , num_75 as num_75_3rdlast \
                    , num_985 as num_985_3rdlast \
                    , num_100 as num_100_3rdlast \
                    , num_unq as num_unq_3rdlast \
                    , CASE WHEN total_secs < 0 THEN 0 ELSE total_secs END as total_secs_clean_3rdlast  \
                    , ABS(total_secs) as abs_total_secs_3rdlast  \
                FROM row_per_user  \
                WHERE row_number = 3'
third_last_row_per_user = spark.sql(third_last_row_per_user_query)
third_last_row_per_user.toPandas().to_csv(data_directory+'test.csv', index=False)

# ===========================================================================================================================

# GETTING THE SUM OF ALL ROWS PER USER
sum_of_rows_query = 'SELECT msno  \
            , COUNT(msno) as row_count \
            , SUM(num_25) as sum_num_25  \
            , SUM(num_50) as sum_num_50  \
            , SUM(num_75) as sum_num_75  \
            , SUM(num_985) as sum_num_985  \
            , SUM(num_100) as sum_num_100  \
            , SUM(num_unq) as sum_num_unq  \
            , SUM(CASE WHEN total_secs < 0 THEN 0 ELSE total_secs END) as sum_total_secs_clean  \
            , SUM(ABS(total_secs)) as sum_abs_total_secs  \
         FROM all_logs  \
         GROUP BY msno'
sum_of_rows = spark.sql(sum_of_rows_query)
sum_of_rows.count()
sum_of_rows.toPandas().to_csv(data_directory+'logs_sum_of_rows_per_user.csv', index=False)

# Sum of last 3 rows per customer for customers with 3 or more rows
sum_of_3_last_rows_query = 'WITH row_per_user as ( \
                    SELECT *  \
                    , row_number() OVER (PARTITION BY msno ORDER BY date DESC) as row_number  \
                    FROM all_logs) ' \
        'SELECT msno  \
            , COUNT(msno) as row_count_3last \
            , SUM(num_25) as sum_num_25_3last  \
            , SUM(num_50) as sum_num_50_3last  \
            , SUM(num_75) as sum_num_75_3last  \
            , SUM(num_985) as sum_num_985_3last  \
            , SUM(num_100) as sum_num_100_3last  \
            , SUM(num_unq) as sum_num_unq_3last  \
            , SUM(CASE WHEN total_secs < 0 THEN 0 ELSE total_secs END) as sum_total_secs_clean_3last  \
            , SUM(ABS(total_secs)) as sum_abs_total_secs_3last  \
         FROM row_per_user  \
         WHERE row_number in (1,2,3) \
         GROUP BY msno \
         HAVING row_count_3last = 3'
sum_of_3_last_rows = spark.sql(sum_of_3_last_rows_query)
sum_of_3_last_rows.show()
