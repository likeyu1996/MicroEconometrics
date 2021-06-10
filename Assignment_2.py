import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.api import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 0
data = pd.read_csv('./Data_2/pubtwins.csv', na_values=np.nan)
# print(data.head())
# 1
'''
mod_1_1 = smf.ols(formula='lwage ~ educ', data=data)
res_1_1 = mod_1_1.fit()
print(res_1_1.summary())
'''

'''
axis_x = data['educ']
axis_y = data['lwage']


axis_x = item_3_2
# axis_y = np.append(value_3_1, value_3_2)
axis_y = value_3_2

sns.set(rc={'figure.figsize': (16, 9)})
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
sns.scatterplot(x='educ', y='lwage', data=data, hue='first', hue_order=[1, 0])
plt.xlabel('educ')
plt.ylabel('lwage')
plt.title('lwage ~ educ for all')
plt.savefig('./Data_2/pic1_1')
plt.close('all')
'''
'''
mod_1_2 = smf.ols(formula='lwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_1_2 = mod_1_2.fit()
print(res_1_2.summary())
sm.graphics.plot_fit(res_1_2, 'educ')
plt.savefig('./Data_2/pic1_2')
plt.close('all')
'''
'''
data['age3'] = data['age'] ** 3
data['age4'] = data['age'] ** 4

mod_1_3 = smf.ols(formula='lwage ~ educ + age + C(female) + C(white)', data=data)
res_1_3 = mod_1_3.fit()
print(res_1_3.summary())
sm.graphics.plot_fit(res_1_3, 'educ')
plt.savefig('./Data_2/pic1_3')
plt.close('all')

mod_1_4 = smf.ols(formula='lwage ~ educ + age + age2 + age3 + C(female) + C(white)', data=data)
res_1_4 = mod_1_4.fit()
print(res_1_4.summary())
sm.graphics.plot_fit(res_1_4, 'educ')
plt.savefig('./Data_2/pic1_4')
plt.close('all')

mod_1_5 = smf.ols(formula='lwage ~ educ + age + age2 + age3 + age4 + C(female) + C(white)', data=data)
res_1_5 = mod_1_5.fit()
print(res_1_5.summary())
sm.graphics.plot_fit(res_1_5, 'educ')
plt.savefig('./Data_2/pic1_5')
plt.close('all')
'''
# 2
'''
sm.graphics.plot_fit(res_1_1, 'educ')
plt.savefig('./Data_2/pic2_1')
plt.close('all')
'''
'''
plt.figure(figsize=(16, 9))
sns.distplot(data['educ'], kde=True, norm_hist=True)
plt.savefig('./Data_2/pic2_2')
'''
'''
print(data['educ'].describe())
'''
'''
data_2_2_1 = data.loc[data['educ'] == 16]
data_2_2_1.reset_index(drop=True, inplace=True)
data_2_2_2 = data.loc[data['educ'] == 12]
data_2_2_2.reset_index(drop=True, inplace=True)
# print(data_2_2_1.mean())
# print(data_2_2_2.mean())
t_list = []
p_2_list = []
for i in data.columns:
    t, p_twoTail = stats.ttest_ind(data_2_2_1.loc[:, i], data_2_2_2.loc[:, i],nan_policy='omit')
    t_list.append(t)
    p_2_list.append(p_twoTail)
result_dict_2_2 = {'Items': data.columns,
                   'V16': data_2_2_1.mean().to_numpy(),
                   'V12': data_2_2_2.mean().to_numpy(),
                   'V16-V12': data_2_2_1.mean().to_numpy() - data_2_2_2.mean().to_numpy(),
                   'T': t_list,
                   'p_2': p_2_list}
result_df_2_2 = pd.DataFrame(result_dict_2_2)
result_df_2_2.to_csv('./Data_2/result_df_2_2.csv', index=False)
'''
# 3
# 剔除educ非整数的数据
'''
ideal_educ = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
data_3 = data.loc[[data.loc[i, 'educ'] in ideal_educ for i in data.index]]
data_3.reset_index(drop=True, inplace=True)
# print(data_3)
'''
'''
mod_3_1 = smf.ols(formula='hrwage ~ C(educ) - 1', data=data_3)
res_3_1 = mod_3_1.fit()
# print(res_3_1.summary())

mod_3_2 = smf.ols(formula='lwage ~ C(educ) - 1', data=data_3)
res_3_2 = mod_3_2.fit()
# print(res_3_2.summary())

array_3_1 = res_3_1.params
item_3_1 = np.array([i[8:-1] for i in array_3_1.index], dtype=float)
value_3_1 = array_3_1.values
dict_3_1 = {'item': item_3_1,
            'value': value_3_1
            }
df_3_1 = pd.DataFrame(dict_3_1)

array_3_2 = res_3_2.params
item_3_2 = np.array([i[8:-1] for i in array_3_2.index], dtype=float)
value_3_2 = array_3_2.values
dict_3_2 = {'item': item_3_2,
            'value': value_3_2
            }
df_3_2 = pd.DataFrame(dict_3_2)


axis_x = item_3_1
# axis_y = np.append(value_3_1, value_3_2)
axis_y = value_3_1
delta_x = np.mean(axis_x)/10.0
delta_y = np.mean(axis_y)/10.0
min_x = np.min(axis_x) - delta_x
max_x = np.max(axis_x) + delta_x
min_y = np.min(axis_y) - delta_y
max_y = np.max(axis_y) + delta_y

sns.set(rc={'figure.figsize': (16, 9)})
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
sns.scatterplot(x='item', y='value', data=df_3_1)
plt.xlabel('Beta')
plt.ylabel('Coff')
plt.title('Hrwage')
plt.savefig('./Data_2/pic3_1')
plt.close('all')

axis_x = item_3_2
# axis_y = np.append(value_3_1, value_3_2)
axis_y = value_3_2
delta_x = np.mean(axis_x)/10.0
delta_y = np.mean(axis_y)/10.0
min_x = np.min(axis_x) - delta_x
max_x = np.max(axis_x) + delta_x
min_y = np.min(axis_y) - delta_y
max_y = np.max(axis_y) + delta_y

sns.set(rc={'figure.figsize': (16, 9)})
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
sns.scatterplot(x='item', y='value', data=df_3_2)
plt.xlabel('Beta')
plt.ylabel('Coff')
plt.title('lwage')
plt.savefig('./Data_2/pic3_2')
plt.close('all')
'''
'''
mod_3_3 = smf.ols(formula='lwage ~ educ + age + age2 + C(female) + C(white)', data=data_3)
res_3_3 = mod_3_3.fit()
print(res_3_3.summary())

mod_3_4 = smf.ols(formula='lwage ~ C(educ) + age + age2 + C(female) + C(white)', data=data_3)
res_3_4 = mod_3_4.fit()
print(res_3_4.summary())
'''
# 4
'''
axis_x = data['educ']
axis_y = data['hrwage']
delta_x = axis_x.mean()/10.0
delta_y = axis_y.mean()/10.0
min_x = axis_x.min() - delta_x
max_x = axis_x.max() + delta_x
min_y = axis_y.min() - delta_y
max_y = axis_y.max() + delta_y

sns.set(rc={'figure.figsize': (16, 9)})
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
sns.scatterplot(x='educ', y='hrwage', data=data, hue='first', hue_order=[1, 0])
plt.xlabel('educ')
plt.ylabel('hrwage')
plt.title('hrwage ~ educ for all')
plt.savefig('./Data_2/pic4_1')
plt.close('all')
'''
'''
mod_4_1 = smf.ols(formula='hrwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_4_1 = mod_4_1.fit()
sm.graphics.plot_regress_exog(res_4_1, 'educ', fig=plt.figure(figsize=(16, 9)))
plt.savefig('./Data_2/pic4_2')

mod_4_2 = smf.ols(formula='lwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_4_2 = mod_4_2.fit()
sm.graphics.plot_regress_exog(res_4_2, 'educ', fig=plt.figure(figsize=(16, 9)))
plt.savefig('./Data_2/pic4_3')
'''
# 5
'''
mod_5_1 = smf.rlm(formula='lwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_5_1 = mod_5_1.fit()
# print(res_5_1.summary())
print(res_5_1.rsquared())
'''
'''
mod_5_1_1 = smf.ols(formula='lwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_5_1_1 = mod_5_1_1.fit()
# print(res_5_1_1.summary())
# res_5_1_2 = res_5_1_1.get_robustcov_results()
# print(res_5_1_2.summary())
# print(sm.stats.cov_white_simple(res_5_1_1))
# print(sm.stats.cov_white_simple(res_5_1_2))
print(sm.stats.spec_white(res_5_1_1.resid, res_5_1_1.model.exog))
# print(sm.stats.spec_white(res_5_1_2.resid, res_5_1_2.model.exog))
'''
'''
# 6
mod_6_1_1 = smf.ols(formula='hrwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_6_1_1 = mod_6_1_1.fit()
resid_6_1_1 = res_6_1_1.resid
# print(resid_6_1_1 ** 2)

mod_6_1_2 = smf.ols(formula='lwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_6_1_2 = mod_6_1_2.fit()
resid_6_1_2 = res_6_1_2.resid
# print(resid_6_1_2 ** 2)

data_6 = data.copy(deep=True)
data_6['res_hr2'] = resid_6_1_1 ** 2
data_6['res_l2'] = resid_6_1_2 ** 2
'''
'''
mod_6_2_1 = smf.ols(formula='res_hr2 ~ educ + age + age2 + C(female) + C(white)', data=data_6)
res_6_2_1 = mod_6_2_1.fit()
print(res_6_2_1.summary())

mod_6_2_2 = smf.ols(formula='res_l2 ~ educ + age + age2 + C(female) + C(white)', data=data_6)
res_6_2_2 = mod_6_2_2.fit()
print(res_6_2_2.summary())
'''
'''
print(sm.stats.het_breuschpagan(resid_6_1_1, res_6_1_1.model.exog, robust=True))
print(sm.stats.het_breuschpagan(resid_6_1_2, res_6_1_2.model.exog, robust=True))
'''
'''
data_6['educ2'] = data_6['educ'] ** 2
mod_6_3_1 = smf.ols(formula='res_hr2 ~ educ + educ2 + age + age2 + C(female) + C(white) + educ*age + educ*C(female) + educ*C(white) +age*C(female) + age*C(white) ', data=data_6)
res_6_3_1 = mod_6_3_1.fit()
print(res_6_3_1.summary())
'''
'''
data_6['educ_age'] = data_6['educ'] * data_6['age']
data_6['educ_female'] = data_6['educ'] * data_6['female']
data_6['educ_white'] = data_6['educ'] * data_6['white']
data_6['age_female'] = data_6['female'] * data_6['age']
data_6['age_white'] = data_6['white'] * data_6['age']
mod_6_3_1_0 = smf.ols(formula='res_hr2 ~ educ + educ2 + age + age2 + C(female) + C(white) + educ_age + educ_female + educ_white +age_female + age_white ', data=data_6)
res_6_3_1_0 = mod_6_3_1_0.fit()
print(res_6_3_1_0.summary())
'''
'''
mod_6_3_2 = smf.ols(formula='res_l2 ~ educ + educ2 + age + age2 + C(female) + C(white) + educ*age + educ*C(female) + educ*C(white) +age*C(female) + age*C(white) ', data=data_6)
res_6_3_2 = mod_6_3_2.fit()
print(res_6_3_2.summary())
'''
'''
mod_6_4_1 = smf.ols(formula='hrwage ~ educ + age + C(female) + C(white)', data=data_6)
res_6_4_1 = mod_6_4_1.fit()
print(sm.stats.het_white(resid_6_1_1, res_6_4_1.model.exog))

mod_6_4_2 = smf.ols(formula='lwage ~ educ + age + C(female) + C(white)', data=data_6)
res_6_4_2 = mod_6_4_2.fit()
print(sm.stats.het_white(resid_6_1_2, res_6_4_2.model.exog))
'''
# 7
'''
print(data)
'''
'''
mod_7_1_1 = smf.ols(formula='hrwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_7_1_1 = mod_7_1_1.fit()
res_7_1_2 = res_7_1_1.get_robustcov_results(cov_type='cluster', groups=data['id'])
print(res_7_1_2.summary())

mod_7_2_1 = smf.ols(formula='lwage ~ educ + age + age2 + C(female) + C(white)', data=data)
res_7_2_1 = mod_7_2_1.fit()
res_7_2_2 = res_7_2_1.get_robustcov_results(cov_type='cluster', groups=data['id'])
print(res_7_2_2.summary())
'''

data_8_1 = data.loc[data['first'] == 1]
data_8_1.reset_index(drop=True, inplace=True)
data_8_2 = data.loc[data['first'] == 0]
data_8_2.reset_index(drop=True, inplace=True)
data_8 = (data_8_1 + data_8_2) / 2
# print(data_8)
'''
mod_8_1 = smf.ols(formula='lwage ~ educ', data=data_8)
res_8_1 = mod_8_1.fit()
print(res_8_1.summary())
'''
mod_8_2 = smf.ols(formula='lwage ~ educ', data=data)
res_8_2 = mod_8_2.fit()
res_8_3 = res_8_2.get_robustcov_results(cov_type='cluster', groups=data['id'])
print(res_8_3.summary())
