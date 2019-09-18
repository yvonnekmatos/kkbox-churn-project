import pyspark
from wd import *
from pyspark.sql.functions import datediff, to_date, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
import pandas as pd

spark = pyspark.sql.SparkSession.builder.getOrCreate()


logs = spark.read.csv(data_directory+'user_logs.csv', header=True, inferSchema=True)

logs2 = spark.read.csv(data_directory+'user_logs_v2.csv', header=True, inferSchema=True)
all_logs = logs.union(logs2)

logs2.show(10)

# logs2.createOrReplaceTempView('all_logs')
all_logs.createOrReplaceTempView('all_logs')

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
last_row_per_user = spark.sql(last_row_per_user_query)
last_row_per_user.show()
last_row_per_user.count()

pd_last_row_per_user = last_row_per_user.toPandas()
pd_last_row_per_user.msno.nunique()

pd_last_row_per_user.describe()

# Total listened
total_listened_query = 'SELECT msno ' \
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

total_listened = spark.sql(total_listened_query)
total_listened.show(5)
total_listened.count()

pd_total_listened = total_listened.toPandas()
pd_total_listened.info()

pd_total_listened.msno.nunique()
pd_total_listened.describe()
pd_last_row_per_user.describe()



pd_total_listened.sum_total_secs_clean.equals(pd_total_listened.sum_abs_total_secs)


pd_total_listened.to_csv(data_directory+'total_listened_subset.csv', index=False)

pd_total_listened.equals(pd_last_row_per_user)
