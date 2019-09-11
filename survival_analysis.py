import pandas as pd
import numpy as np
from wd import *
import lifelines as ll
import seaborn as sns
import matplotlib.pyplot as plt

churn_months = pd.read_csv(data_directory+'survival_data/survival_2015_after.csv')

churn_months.month.max()

churn_months.is_churn_final.mean()
churn_months.info()

churn_months = churn_months[churn_months.is_churn_final.notnull()]

sns.distplot(churn_months.month);


T = churn_months.month
E = churn_months.is_churn_final

kmf = ll.KaplanMeierFitter()
kmf.fit(T, E)

kmf.survival_function_
kmf.cumulative_density_
kmf.median_
kmf.plot();


naf = ll.NelsonAalenFitter()
naf.fit(T, E)

naf.plot();
naf.cumulative_hazard_
