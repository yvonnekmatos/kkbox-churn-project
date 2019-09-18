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

sf = kmf.survival_function_.reset_index()
kmf.cumulative_density_
kmf.median_
kmf.plot();
sf


plt.step(sf.timeline, sf.KM_estimate*100, where='post', color='#1e488f')
plt.xlim(0, 27.5)
plt.savefig(data_directory+'data_viz/km_plot2.png', transparent=True, dpi=2000)



naf = ll.NelsonAalenFitter()
naf.fit(T, E)

naf.plot();
ch = naf.cumulative_hazard_.reset_index()
ch_sub = ch[ch.timeline.isin([6, 12, 18, 24])]
ch_sub
