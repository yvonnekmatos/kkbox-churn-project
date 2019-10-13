import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wd import *

# VIZUALIZE DATA FOR PRESENTATION 

def city_groups(x):
    if x in ['21.0', '12.0', '8.0']:
        return 1 # Southwest
    elif x in ['10.0', '5.0', '4.0', '3.0', '6.0', '15.0']:
        return 2 # Midwest
    elif x in ['22.0', '9.0', '13.0', '11.0', '18.0']:
        return 3 # Southeast
    elif x in ['16.0', '14.0', '19.0', '20.0']:
        return 4 # Northwest
    elif x in ['7.0', '17.0']:
        return 5 # Northeast
    else:
        return 6 # South


mem_target = pd.read_csv(data_directory+'clean_members_w_target.csv')
mem_target.shape
mem_target['city'] = mem_target.city.astype(str).fillna('999')
mem_target['registered_via'] = mem_target.registered_via.astype(str).fillna('999')
mem_target = mem_target.drop('registration_init_time', axis=1)

# REGISTRATION METHOD PLOT
via = pd.DataFrame(mem_target.groupby('registered_via').is_churn_final.mean()).reset_index()
via = via.sort_values(by='is_churn_final', ascending=False)
via
plt.bar(x=via.registered_via.values, height=via.is_churn_final.values*100, color='#1e488f')
plt.ylim([0, 40])
plt.savefig(data_directory+'data_viz/registered_via_bar.png', transparent=True);

# AGE PLOT
mem_target_no_null_age = mem_target[(mem_target.bd != 999)]
age = pd.DataFrame(mem_target_no_null_age.groupby('bd').is_churn_final.mean()).reset_index()
age = age[(age.bd >= 15) & (age.bd <= 55)]
age = age.sort_values(by='bd', ascending=False)
age
barplot = plt.bar(x=age.bd.values, height=age.is_churn_final.values*100, color='#1e488f')
plt.ylim([0, 40])
plt.savefig(data_directory+'data_viz/age_bar.png', transparent=True);


# EXPLORE CUSTOMER SEGMENTS BY CHURN RATE
mem_target['is_24_or_older'] = mem_target.apply(lambda row: 1 if row.bd >=24 else 0, axis=1)
mem_target['city_groups'] = mem_target.city.apply(city_groups)
mem_target.info()

segs = pd.DataFrame(mem_target.groupby(['is_24_or_older', 'city_groups', 'registered_via']).is_churn_final.mean())\
.sort_values(by='is_churn_final', ascending=False).reset_index()
segs = segs.sort_values(by='is_churn_final', ascending=False).reset_index()
segs

plt.bar(x=segs.index.values, height=segs.is_churn_final.values, color='#1e488f')
plt.ylim([0, 0.4]);

# EXPLORE GROUPING CITIES BY CHURN RATE
city = pd.DataFrame(mem_target.groupby('city').is_churn_final.mean()).reset_index()
city = city.sort_values(by='is_churn_final', ascending=False)
city
plt.bar(x=city.city.values, height=city.is_churn_final.values*100, color='#1e488f');


cityg = pd.DataFrame(mem_target.groupby('city_groups').is_churn_final.mean()).reset_index()
cityg = cityg.sort_values(by='is_churn_final', ascending=False)
cityg['city_groups'] = cityg.city_groups.astype(str)

plt.bar(x=cityg.city_groups.values, height=cityg.is_churn_final.values*100, color='#1e488f')
plt.ylim([0, 40])
plt.savefig(data_directory+'data_viz/city_group.png', transparent=True);
