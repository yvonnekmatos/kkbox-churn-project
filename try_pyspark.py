import pyspark
from wd import *
from pyspark.sql.functions import datediff, to_date, lit


spark = pyspark.sql.SparkSession.builder.getOrCreate()

logs = spark.read.csv(data_directory+'user_logs.csv', header=True, inferSchema=True)
# logs.count()
# logs.printSchema()
#
# logs.show(5)

logs2 = spark.read.csv(data_directory+'user_logs_v2.csv', header=True, inferSchema=True)
# logs2.show(5)
# logs2.count()

all_logs = logs.union(logs2)
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
