import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.api import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns


# 1
# 读取数据
data = pd.read_csv('./Data_1/dataset.csv')
# print(data)
# 删除所有非工薪者
data = data.loc[data.loc[:, 'paidwork'] == 1].reset_index(drop=True)
# print(data)
# 剩余数据行数
# print('删除所有非工薪者数据后，剩余数据行数为{0}'.format(len(data.index.to_numpy())))
# 添加变量school=yprim+ysec和wage=e^(lwage)
data.loc[:, 'school'] = data.loc[:, 'yprim']+data.loc[:, 'ysec']
data.loc[:, 'wage'] = np.exp(data.loc[:, 'lwage'])
# 删除列'urban', 'unearn', 'househ', 'amtland', 'unearnx'
data = data.drop(['urban', 'unearn', 'househ', 'amtland', 'unearnx'], axis=1)
# print(data)
'''
# 2
# 求lwage和wage的均值和中位数
mean_list = [np.mean(data.loc[:, 'wage']), np.mean(data.loc[:, 'lwage']), np.log(np.mean(data.loc[:, 'wage']))]
median_list = [np.median(data.loc[:, 'wage']), np.median(data.loc[:, 'lwage']), np.log(np.median(data.loc[:, 'wage']))]
df_2 = pd.DataFrame({'mean': mean_list, 'median': median_list}, index=['wage', 'lwage', 'ln(wage)'])
# print(df_2)
# print('mean(lwage)不等于ln(mean(wage)), median(lwage)约等于ln(median(wage))')
# 3
mod_3 = smf.ols(formula='wage ~ school', data=data, hasconst=False)
res_3 = mod_3.fit()
print(res_3.summary())
# 4
mod_4 = smf.ols(formula='lwage ~ school', data=data)
res_4 = mod_4.fit()
print(res_4.summary())

# 5
mod_5 = smf.ols(formula='lwage ~ age + school + C(chinese) + C(indian) + C(men)', data=data)
res_5 = mod_5.fit()
print(res_5.summary())

sns.set(rc={'figure.figsize': (16, 9)})
# sns.lineplot(x='school', y='lwage', data=data, marker='*')
sns.jointplot(x='school', y='lwage', data=data, kind='reg')
plt.xlabel('school')
plt.ylabel('lwage')
plt.show()
'''
'''
# 7
table_1 = anova_lm(res_5)
print(table_1)
'''

# 8

data_men = data.loc[data.loc[:, 'men'] == 1].reset_index(drop=True)
data_women = data.loc[data.loc[:, 'men'] == 0].reset_index(drop=True)
# print(data_men)
# print(data_women)
mod_men = smf.ols(formula='lwage ~ age + school + C(chinese) + C(indian) + C(men)', data=data_men)
res_men = mod_men.fit()
# print(res_men.summary())
print(res_men.ssr)
mod_women = smf.ols(formula='lwage ~ age + school + C(chinese) + C(indian) + C(men)', data=data_women)
res_women = mod_women.fit()
# print(res_women.summary())
print(res_women.ssr)
mod_c = smf.ols(formula='lwage ~ age + school + C(chinese) + C(indian)', data=data)
res_c = mod_c.fit()
print(res_c.ssr)

