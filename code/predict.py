#!/usr/bin/env python
# coding: utf-8

# In[437]:


import pandas as pd
import numpy as np


# In[438]:


dataframe=pd.read_excel('rollingsales_brooklyn.xls',header=4)


# In[439]:


dataframe.head()


# In[440]:


dataframe.shape


# In[441]:


dataframe.isnull().sum(axis=0)


# In[442]:


dataframe.shape


# In[443]:


dataframe=dataframe.drop(columns=['EASE-MENT','APARTMENT NUMBER'],axis=1)


# In[444]:


dataframe.dtypes


# In[445]:


dataframe['SALE PRICE'].value_counts()


# In[446]:


dataframe=dataframe[dataframe['SALE PRICE']!=0]


# In[447]:


dataframe.shape


# In[448]:


dataframe['YEAR BUILT']=dataframe['YEAR BUILT'].replace('NaN',np.nan)


# In[449]:


dataframe['YEAR BUILT'].isnull().sum(axis=0)


# In[450]:


dataframe['YEAR BUILT'].unique()


# In[451]:


dataframe['YEAR BUILT'] = 2019-dataframe['YEAR BUILT']


# In[452]:


from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan,strategy='mean')


# In[453]:


dataframe['YEAR BUILT']=dataframe['YEAR BUILT'].fillna(dataframe['YEAR BUILT'].mean())


# In[454]:


dataframe['YEAR BUILT']=dataframe['YEAR BUILT'].astype('int')


# In[455]:


dataframe['YEAR BUILT'].unique()


# In[456]:


dataframe['YEAR BUILT'].isnull().sum(axis=0)


# In[457]:


dataframe.head()


# In[458]:


dataframe['BUILDING CLASS AT PRESENT'].unique()


# In[459]:


dataframe['BUILDING CLASS AT PRESENT']=dataframe['BUILDING CLASS AT PRESENT'].fillna(dataframe['BUILDING CLASS AT PRESENT'].mode()[0])


# In[460]:


dataframe['BUILDING CLASS AT PRESENT'].unique()


# In[461]:


dataframe['BUILDING CLASS AT PRESENT'].isnull().sum(axis=0)


# In[462]:


dataframe['TAX CLASS AT PRESENT']=dataframe['TAX CLASS AT PRESENT'].fillna(dataframe['TAX CLASS AT PRESENT'].mode()[0])


# In[463]:


dataframe['TAX CLASS AT PRESENT'].isnull().sum(axis=0)


# In[464]:


dataframe.isnull().sum(axis=0)


# In[465]:


dataframe.dtypes


# In[466]:


dataframe['LAND SQUARE FEET'].isnull().sum(axis=0)


# In[467]:


dataframe.shape


# In[468]:


dataframe['LAND SQUARE FEET']=dataframe['LAND SQUARE FEET'].replace(0.0,np.nan)


# In[469]:


dataframe['LAND SQUARE FEET']=dataframe['LAND SQUARE FEET'].replace('NaN',np.nan)


# In[470]:


dataframe['LAND SQUARE FEET']=dataframe['LAND SQUARE FEET'].fillna(dataframe['LAND SQUARE FEET'].mean())


# In[471]:


dataframe['GROSS SQUARE FEET'].isnull().sum(axis=0)


# In[472]:


dataframe['GROSS SQUARE FEET']=dataframe['GROSS SQUARE FEET'].replace(0.0,np.nan)
dataframe['GROSS SQUARE FEET']=dataframe['GROSS SQUARE FEET'].replace('NaN',np.nan)


# In[473]:


dataframe['GROSS SQUARE FEET']=dataframe['GROSS SQUARE FEET'].fillna(dataframe['GROSS SQUARE FEET'].mean())


# In[474]:


dataframe['RESIDENTIAL UNITS'].value_counts()


# In[475]:


dataframe['RESIDENTIAL UNITS']=dataframe['RESIDENTIAL UNITS'].replace('NaN',np.nan)


# In[476]:


dataframe['RESIDENTIAL UNITS']=dataframe['RESIDENTIAL UNITS'].fillna(dataframe['RESIDENTIAL UNITS'].mean())


# In[477]:


dataframe['COMMERCIAL UNITS'].value_counts()


# In[478]:


dataframe['COMMERCIAL UNITS']=dataframe['COMMERCIAL UNITS'].replace('NaN',np.nan)


# In[479]:


dataframe['COMMERCIAL UNITS']=dataframe['COMMERCIAL UNITS'].fillna(dataframe['COMMERCIAL UNITS'].mean())


# In[480]:


dataframe['TOTAL UNITS']=dataframe['COMMERCIAL UNITS']+dataframe['RESIDENTIAL UNITS']


# In[481]:


dataframe.isnull().sum(axis=0)


# In[482]:


dataframe['RESIDENTIAL UNITS']=dataframe['RESIDENTIAL UNITS'].astype('int')
dataframe['COMMERCIAL UNITS']=dataframe['COMMERCIAL UNITS'].astype('int')
dataframe['TOTAL UNITS']=dataframe['TOTAL UNITS'].astype('int')


# In[483]:


dataframe.dtypes


# In[484]:


dataframe.head()


# In[485]:


dataframe=dataframe.drop(columns=['ADDRESS'],axis=1)


# In[486]:


dataframe.shape


# In[487]:


dataframe.NEIGHBORHOOD.value_counts()


# In[488]:


dataframe['BUILDING CLASS CATEGORY'].value_counts()


# In[489]:


dataframe['BOROUGH']=dataframe['BOROUGH'].astype('category')
dataframe['BOROUGH']=dataframe['BOROUGH'].cat.codes

dataframe['NEIGHBORHOOD']=dataframe['NEIGHBORHOOD'].astype('category')
dataframe['NEIGHBORHOOD']=dataframe['NEIGHBORHOOD'].cat.codes

dataframe['BUILDING CLASS AT PRESENT']=dataframe['BUILDING CLASS AT PRESENT'].astype('category')
dataframe['BUILDING CLASS AT PRESENT']=dataframe['BUILDING CLASS AT PRESENT'].cat.codes

dataframe['BUILDING CLASS AT TIME OF SALE']=dataframe['BUILDING CLASS AT TIME OF SALE'].astype('category')
dataframe['BUILDING CLASS AT TIME OF SALE']=dataframe['BUILDING CLASS AT TIME OF SALE'].cat.codes

dataframe['BUILDING CLASS CATEGORY']=dataframe['BUILDING CLASS CATEGORY'].astype('category')
dataframe['BUILDING CLASS CATEGORY']=dataframe['BUILDING CLASS CATEGORY'].cat.codes

dataframe['TAX CLASS AT PRESENT']=dataframe['TAX CLASS AT PRESENT'].astype('category')
dataframe['TAX CLASS AT PRESENT']=dataframe['TAX CLASS AT PRESENT'].cat.codes

dataframe['ZIP CODE']=dataframe['ZIP CODE'].astype('category')
dataframe['ZIP CODE']=dataframe['ZIP CODE'].cat.codes


# In[490]:


dataframe.head()


# In[491]:


dataframe.dtypes


# In[492]:


dataframe['SALE DATE']=dataframe['SALE DATE'].map(lambda x:x.month)


# In[493]:


dataframe['SALE DATE'].value_counts()


# In[494]:


dataframe.head()


# In[495]:


dataframe.shape


# In[496]:


dataframe.dtypes


# In[497]:


dataframe['YEAR BUILT'].value_counts()


# In[498]:


# column_names = dataframe_continuous.columns


# In[499]:




# Apply the power transform to the data
dataframe[['BLOCK', 'LOT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 'LAND SQUARE FEET', 'GROSS SQUARE FEET']] = (dataframe[['BLOCK', 'LOT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 'LAND SQUARE FEET', 'GROSS SQUARE FEET']]**(1/2))
# dataframe_continuous.columns = column_names


# In[500]:


dataframe=dataframe[dataframe['TOTAL UNITS']!=0]


# In[501]:


# dataframe['AVERAGESQUARE'] = dataframe['LAND SQUARE FEET'] / dataframe['TOTAL UNITS']


# In[502]:


dataframe.head()


# In[503]:


features = dataframe.drop(columns='SALE PRICE')
target = dataframe[['SALE PRICE']]


# In[504]:


features_name = features.columns


# In[505]:


features.head()


# In[506]:


# features['AVERAGESQUARE'].value_counts()


# In[507]:


target.head()


# In[508]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
features = pd.DataFrame(scalar.fit_transform(features))
features.columns = features_name


# In[509]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[510]:


# Load library
from sklearn.linear_model import RidgeCV

# Create ridge regression with three alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Fit the linear regression
model_cv = regr_cv.fit(x_train, y_train)

# View coefficients
model_cv.coef_


# In[511]:


model_cv.alpha_


# In[512]:


# Apply the model we created using the training data 
# to the test data, and calculate the RSS.
((y_test.values - model_cv.predict(x_test)) **2).sum()


# In[513]:


model_cv.predict(x_test)


# In[514]:


# For MSE:
mse = np.mean((y_train.values - model_cv.predict(x_train)) **2)
mse


# In[515]:


rmse = mse**(1/2)
rmse


# In[516]:


# For MSE:
mse = np.mean((y_test.values - model_cv.predict(x_test)) **2)
mse


# In[517]:


rmse = mse**(1/2)
rmse


# In[518]:


# For R2
from sklearn.metrics import r2_score
r2_score(y_train.values, model_cv.predict(x_train)) 


# In[519]:


r2 = r2_score(y_test.values, model_cv.predict(x_test)) 
r2


# In[520]:


# To calculate adjusted R2
adjusted_r_squared = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
adjusted_r_squared


# In[521]:


import numpy as np

from sklearn.linear_model import LassoCV
from yellowbrick.regressor import AlphaSelection

# Create a list of alphas to cross-validate against
alphas = np.logspace(-10, 1, 400)

# Instantiate the linear model and visualizer
model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model)
visualizer.fit(x_train, y_train)
visualizer.show()


# In[522]:


# Load library
from sklearn.linear_model import Lasso

# Load data
features = x_train
target = y_train

# Create lasso regression with alpha value
regression = Lasso(alpha=9.385)

# Fit the linear regression
model = regression.fit(features, target)


# In[523]:


model.score(x_test, y_test)


# In[524]:


# Apply the model we created using the training data 
# to the test data, and calculate the RSS.
((y_test.values - model.predict(x_test)) **2).sum()


# In[525]:


# For MSE:
mse = np.mean((y_test.values - model.predict(x_test)) **2)
mse


# In[526]:


rmse = mse**(1/2)
rmse


# In[ ]:





# In[70]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(lgb_vars, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[71]:


import lightgbm


# In[72]:


lgb_model=lightgbm.LGBMRegressor(
boosting_type='gbdt',
objective='regression',
    num_leaves=9,
    learning_rate=0.28,
    max_depth=4,
    metric='rmse',
    lambda_l1=11,
    lambda_l2=10,
    alpha=10,
    reg_alpha=9,
    reg_lambda=8,
    n_estimators=600,
     min_child_weight=6,
    bagging_fraction=0.8,
    max_bin=60
)


# In[73]:


lgb_model.fit(x_train,y_train)


# In[74]:


import sklearn.metrics as metrics
lgb_train = lgb_model.predict(x_train)
lgb_test = lgb_model.predict(x_test)


# In[75]:


lgb_model.score(x_train,y_train)


# In[76]:


lgb_model.score(x_test,y_test)


# In[77]:


mse_train=np.mean((y_train.values - lgb_train) **2)
mse_train**0.5


# In[78]:


mse_test=np.mean((y_test.values - lgb_test) **2)
mse_test**0.5


# In[79]:


import xgboost 


# In[80]:


xgb_model=xgboost.XGBRegressor(
    base_score=0.7,
    eta=0.2,
    gamma=3,
    learning_rate=0.35,
    max_depth=4,
    min_child_weight=6,
    n_estimators=300,
    reg_alpha=0.2,
    reg_lambda=0.9,
    scale_pos_weight=2
)


# In[81]:


xgb_model.fit(x_train,y_train)


# In[82]:


xgb_train = xgb_model.predict(x_train)
xgb_test = xgb_model.predict(x_test)


# In[83]:


from sklearn.metrics import r2_score
r2_score(y_train.values, xgb_train) 


# In[84]:


r2_score(y_test.values, xgb_test)


# In[85]:


xgb_train_mse=np.mean((y_train.values-xgb_train)**2)
xgb_train_mse


# In[86]:


xgb_train_mse**0.5


# In[87]:


xgb_test_mse=np.mean((y_test.values - xgb_test) **2)
xgb_test_mse


# In[88]:


xgb_test_mse**0.5


# # XGB-SHAP

# In[89]:


import shap


# In[90]:


xgb_shap_explainer = shap.TreeExplainer(xgb_model)


# In[92]:


xgb_shap_vals_train = xgb_shap_explainer.shap_values(x_train,check_additivity=False)


# In[93]:


xgb_shap_vals_test = xgb_shap_explainer.shap_values(x_test)


# In[94]:


shap.initjs()
shap.force_plot(xgb_shap_explainer.expected_value, xgb_shap_vals_train[0,:], x_train.iloc[0,:])


# In[95]:


shap.summary_plot(xgb_shap_vals_train, x_train)


# In[96]:


shap.summary_plot(xgb_shap_vals_test, x_test)


# In[97]:


lgb_shap_explainer = shap.TreeExplainer(lgb_model)


# In[98]:


lgb_shap_vals_train = lgb_shap_explainer.shap_values(x_train)
lgb_shap_vals_test = lgb_shap_explainer.shap_values(x_test)


# In[99]:


shap.initjs()
shap.force_plot(lgb_shap_explainer.expected_value, lgb_shap_vals_train[0,:], x_train.iloc[0,:])


# In[100]:


shap.summary_plot(lgb_shap_vals_train, x_train)


# In[101]:


shap.summary_plot(lgb_shap_vals_test, x_test)


# In[ ]:





# In[455]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_vars, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[454]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=1)


# In[456]:


rf_regr = RandomForestRegressor(max_depth = 100, random_state=0,n_estimators=5
                               )


# In[457]:


rf_regr_model = rf_regr.fit(x_train, y_train)


# In[458]:


rf_regr_model


# In[493]:


train_pr = rf_regr_model.predict(x_train)


# In[494]:


fare_preds = rf_regr_model.predict(x_test)


# In[495]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, fare_preds)


# In[496]:


mean_squared_error(y_train, train_pr)


# In[497]:


#RMSE
from math import sqrt
sqrt(mean_squared_error(y_test, fare_preds))


# In[498]:


sqrt(mean_squared_error(y_train, train_pr))


# In[499]:


from sklearn.metrics import r2_score


# In[500]:


r2_score(y_train, train_pr)


# In[501]:


r2_score(y_test, fare_preds)


# In[86]:


import matplotlib.pyplot as plt
plt.scatter(fare_preds, y_test)


# In[514]:


rfc_param_grid = {
    'max_depth': [2, 5, 30, 60, 100
                 ],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [5, 8, 10],
    'n_estimators': [5, 10],
    #'criterion' : ['gini', 'entropy']
    #,min_impurity_decrease : [???]
    #,min_impurity_split : [???]
    #,class_weight : [???]
}


# In[515]:


from sklearn.model_selection import GridSearchCV
rfc_gs = GridSearchCV(estimator=rf_regr, param_grid=rfc_param_grid, cv= 5,
                     #scoring methods also include: accuracy, balanced accuracy, average precision, f1, etc.  see documentation
                     )


# In[516]:


dfs = [x_train, x_val]
dfsy = [y_train, y_val]
x_train_rf = pd.concat(dfs)
y_train_rf = pd.concat(dfsy)


# In[517]:


rfc_gs.fit(x_train_rf, y_train_rf)


# In[518]:


rfc_gs.best_params_


# In[519]:


train_pr_grid = rfc_gs.predict(x_train)


# In[520]:


fare_preds_grid = rfc_gs.predict(x_test)


# In[521]:


mean_squared_error(y_test, fare_preds_grid)


# In[522]:


sqrt(mean_squared_error(y_test, fare_preds_grid))


# In[523]:


sqrt(mean_squared_error(y_train, train_pr_grid))


# In[524]:


r2_score(y_train, train_pr_grid)


# In[525]:


r2_score(y_test, fare_preds_grid)


# In[ ]:





# In[455]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_vars, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[454]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=1)


# In[456]:


rf_regr = RandomForestRegressor(max_depth = 100, random_state=0,n_estimators=5
                               )


# In[457]:


rf_regr_model = rf_regr.fit(x_train, y_train)


# In[458]:


rf_regr_model


# In[493]:


train_pr = rf_regr_model.predict(x_train)


# In[494]:


fare_preds = rf_regr_model.predict(x_test)


# In[495]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, fare_preds)


# In[496]:


mean_squared_error(y_train, train_pr)


# In[497]:


#RMSE
from math import sqrt
sqrt(mean_squared_error(y_test, fare_preds))


# In[498]:


sqrt(mean_squared_error(y_train, train_pr))


# In[499]:


from sklearn.metrics import r2_score


# In[500]:


r2_score(y_train, train_pr)


# In[501]:


r2_score(y_test, fare_preds)


# In[86]:


import matplotlib.pyplot as plt
plt.scatter(fare_preds, y_test)


# In[514]:


rfc_param_grid = {
    'max_depth': [2, 5, 30, 60, 100
                 ],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [5, 8, 10],
    'n_estimators': [5, 10],
    #'criterion' : ['gini', 'entropy']
    #,min_impurity_decrease : [???]
    #,min_impurity_split : [???]
    #,class_weight : [???]
}


# In[515]:


from sklearn.model_selection import GridSearchCV
rfc_gs = GridSearchCV(estimator=rf_regr, param_grid=rfc_param_grid, cv= 5,
                     #scoring methods also include: accuracy, balanced accuracy, average precision, f1, etc.  see documentation
                     )


# In[516]:


dfs = [x_train, x_val]
dfsy = [y_train, y_val]
x_train_rf = pd.concat(dfs)
y_train_rf = pd.concat(dfsy)


# In[517]:


rfc_gs.fit(x_train_rf, y_train_rf)


# In[518]:


rfc_gs.best_params_


# In[519]:


train_pr_grid = rfc_gs.predict(x_train)


# In[520]:


fare_preds_grid = rfc_gs.predict(x_test)


# In[521]:


mean_squared_error(y_test, fare_preds_grid)


# In[522]:


sqrt(mean_squared_error(y_test, fare_preds_grid))


# In[523]:


sqrt(mean_squared_error(y_train, train_pr_grid))


# In[524]:


r2_score(y_train, train_pr_grid)


# In[525]:


r2_score(y_test, fare_preds_grid)


# In[1]:


import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# In[2]:


df=pd.read_csv('new_final.csv')


# In[3]:


df.head()


# In[4]:


df=df.drop(columns=['Cid'],axis=1)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df=df.drop(columns=['constructionTime'],axis=1)


# In[9]:


df=df.drop(columns=['DOM'],axis=1)


# In[10]:


df.shape


# In[11]:


df.head()


# In[12]:


df['tradeTime'] = pd.to_datetime(df['tradeTime'],format="%m/%d/%y")


# In[13]:


df.head()


# In[14]:


df['tradeTime']=df['tradeTime'].map(lambda x: 100*x.year+x.month)


# In[15]:


df.head()


# In[16]:


df.isnull().sum(axis=0)


# In[17]:


df.dtypes


# In[18]:


df['tradeTime1']=df['tradeTime'].copy(deep = True)


# In[19]:


df['tradeTime1'].head()


# In[20]:


df=df[df['tradeTime1']>=201600]


# In[21]:


df['tradeTime1'].head()


# In[22]:


df.isnull().sum(axis=0)


# In[23]:


df.shape


# In[24]:


df.fillna(value = {'communityAverage': df['communityAverage'].mean()},       
                              inplace = True
          )


# In[25]:


df.fillna(value = {'buildingType': df['buildingType'].mode()[0]},       
                              inplace = True
          )


# In[26]:


df['buildingType'] = df['buildingType'].apply(int)


# In[27]:


df.dtypes


# In[28]:


df['elevator'] = df['elevator'].apply(int)


# In[29]:


df['fiveYearsProperty'] = df['fiveYearsProperty'].apply(int)


# In[30]:


df['subway'] = df['subway'].apply(int)


# In[31]:


df=df.drop(columns=['totalPrice'],axis=1)


# In[32]:


df['tradeyear']=df.tradeTime.astype(str).str[0:4]


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[34]:


df_year=df.groupby('tradeyear').price.mean()
sns.lineplot(data=df_year)


# In[35]:


df.tradeyear.value_counts()


# In[36]:


df_2016=df.loc[df.tradeyear=='2016',:]


# In[37]:


plt.figure(figsize=(5,5))
sns.scatterplot(x=df_2016['Lng'],y=df_2016['Lat'],hue=df_2016['price'])


# In[38]:


sns.distplot(a=df_2016['square'],kde=False)


# In[39]:


sns.scatterplot(x=df_2016['square'],y=df_2016['followers'])


# In[40]:


plt.figure(figsize=(30,3))
ConstructionTime_price=df_2016.groupby('renovationCondition').price.mean()
sns.barplot(x=ConstructionTime_price.index,y=ConstructionTime_price)


# In[41]:


from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from numpy import exp
from numpy import log


# In[42]:


qqplot(df['square'], line='s')
pyplot.show()


# In[43]:


qqplot(df['ladderRatio'], line='s')
pyplot.show()


# In[44]:


t = log(df['communityAverage'])
df['communityAverage'] = t


# In[45]:


qqplot(df['communityAverage'], line='s')
pyplot.show()


# In[46]:


df['followers'] = (df[['followers']]**(1/2))


# In[47]:


qqplot(df['followers'], line='s')
pyplot.show()


# In[48]:


df.head()


# In[49]:


target = df[['price']]


# In[50]:


features = df.drop(columns='price')


# In[51]:


features_name = features.columns


# In[52]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
features = pd.DataFrame(scalar.fit_transform(features))
features.columns = features_name


# In[53]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[54]:


# Load library
from sklearn.linear_model import RidgeCV

# Create ridge regression with three alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Fit the linear regression
model_cv = regr_cv.fit(x_train, y_train)

# View coefficients
model_cv.coef_


# In[55]:


model_cv.alpha_


# In[56]:


# For MSE:
mse = np.mean((y_train.values - model_cv.predict(x_train)) **2)


# In[57]:


# For RMSE:
rmse = (mse**(1/2))
rmse


# In[58]:


# For MSE:
mse = np.mean((y_test.values - model_cv.predict(x_test)) **2)


# In[59]:


# For RMSE:
rmse = (mse**(1/2))
rmse


# In[60]:


# For R2
from sklearn.metrics import r2_score
r2_score(y_train.values, model_cv.predict(x_train)) 


# In[61]:


from sklearn.metrics import r2_score
r2_score(y_test.values, model_cv.predict(x_test)) 


# In[62]:


# As we discussed in our lecture, alpha is a hyperparemter.  To find the best hyperparameter, we should use grid-search.
# We can do this for LASSO with LASSO CV
from sklearn.linear_model import LassoCV


# In[63]:


lasso_cv = LassoCV(alphas = [0.01, 0.1, 0.5, 1, 5, 10])


# In[64]:


# Fit the linear regression
model_cv = lasso_cv.fit(x_train, y_train)


# In[72]:


# For MSE:
mse = np.mean((y_train.values - model_cv.predict(x_train)) **2)


# In[73]:


# For RMSE:
rmse = (mse**(1/2))
rmse


# In[74]:


# For MSE:
mse = np.mean((y_test.values - model_cv.predict(x_test)) **2)


# In[75]:


# For RMSE:
rmse = (mse**(1/2))
rmse


# In[76]:


# For R2
from sklearn.metrics import r2_score
r2_score(y_train.values, model_cv.predict(x_train)) 


# In[77]:


# For R2
from sklearn.metrics import r2_score
r2_score(y_test.values, model_cv.predict(x_test)) 


# In[ ]:





# In[ ]:





# In[45]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(lgb_vars, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[46]:


import lightgbm
lgb_model=lightgbm.LGBMRegressor(
boosting_type='gbdt',
objective='regression',
    num_leaves=9,
    learning_rate=0.28,
    max_depth=4,
    metric='rmse',
    lambda_l1=11,
    lambda_l2=10,
    alpha=10,
    reg_alpha=9,
    reg_lambda=8,
    n_estimators=600,
     min_child_weight=6,
    bagging_fraction=0.8,
    max_bin=60
)


# In[47]:


df.dtypes


# In[48]:


lgb_model.fit(x_train,y_train)


# In[49]:


import sklearn.metrics as metrics
lgb_train = lgb_model.predict(x_train)
lgb_test = lgb_model.predict(x_test)


# In[50]:


lgb_model.score(x_train,y_train)


# In[51]:


lgb_model.score(x_test,y_test)


# In[52]:


mse_train=np.mean((y_train.values - lgb_train) **2)
mse_train**0.5


# In[53]:


mse_test=np.mean((y_test.values - lgb_test) **2)
mse_test**0.5


# In[54]:


import xgboost 
xgb_model=xgboost.XGBRegressor(
    base_score=0.7,
    eta=0.2,
    gamma=3,
    learning_rate=0.35,
    max_depth=4,
    min_child_weight=6,
    n_estimators=300,
    reg_alpha=0.2,
    reg_lambda=0.9,
    scale_pos_weight=2
)


# In[55]:


xgb_model.fit(x_train,y_train)


# In[56]:


xgb_train = xgb_model.predict(x_train)
xgb_test = xgb_model.predict(x_test)


# In[57]:


from sklearn.metrics import r2_score
r2_score(y_train.values, xgb_train) 


# In[58]:


r2_score(y_test.values, xgb_test)


# In[59]:


xgb_train_mse=np.mean((y_train.values-xgb_train)**2)
xgb_train_mse


# In[60]:


xgb_train_mse**0.5


# In[61]:


xgb_test_mse=np.mean((y_test.values - xgb_test) **2)
xgb_test_mse


# In[62]:


xgb_test_mse**0.5


# In[63]:


import shap


# # XGBoost-SHAP

# In[64]:


xgb_shap_explainer = shap.TreeExplainer(xgb_model)


# In[65]:


xgb_shap_vals_train = xgb_shap_explainer.shap_values(x_train)


# In[66]:


xgb_shap_vals_test = xgb_shap_explainer.shap_values(x_test)


# In[67]:


shap.initjs()
shap.force_plot(xgb_shap_explainer.expected_value, xgb_shap_vals_train[0,:], x_train.iloc[0,:])


# In[68]:


shap.initjs()
shap.dependence_plot("square", xgb_shap_vals_test, x_test)


# In[69]:


shap.summary_plot(xgb_shap_vals_train, x_train)


# In[70]:


shap.summary_plot(xgb_shap_vals_test, x_test)


# # LightGBM-SHAP

# In[71]:


lgb_shap_explainer = shap.TreeExplainer(lgb_model)


# In[72]:


lgb_shap_vals_train = lgb_shap_explainer.shap_values(x_train)
lgb_shap_vals_test = lgb_shap_explainer.shap_values(x_test)


# In[73]:


shap.initjs()
shap.force_plot(lgb_shap_explainer.expected_value, lgb_shap_vals_train[0,:], x_train.iloc[0,:])


# In[74]:


shap.initjs()
shap.dependence_plot("square", lgb_shap_vals_test, x_test)


# In[75]:


shap.summary_plot(lgb_shap_vals_train, x_train)


# In[76]:


shap.summary_plot(lgb_shap_vals_test, x_test)


# In[78]:


deploy=lgb_vars.head()


# In[79]:


deploy


# In[85]:


deploy=deploy[0:1]


# In[86]:


deploy


# In[87]:


deploy.to_excel("deploy.xlsx")


# In[91]:


line_vars=pd.read_csv('deploy.csv')


# In[92]:


line_vars


# In[93]:


price_pre=lgb_model.predict(line_vars)


# In[95]:


price_pre


# In[100]:


import numpy as np
import matplotlib.pyplot as plt
 
x=[1,2,3,4,5,6,7,8,9,10,11,12]
y=price_pre
plt.figure()
plt.plot(x,y)
plt.show()


# In[101]:


xgb_pre=xgb_model.predict(line_vars)


# In[102]:


xgb_pre


# In[ ]:





# In[342]:


from sklearn.ensemble import RandomForestRegressor


# In[345]:


x_vars =x_vars.drop(columns=['followers'],axis=1)


# In[346]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_vars, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)


# In[347]:


rf_regr = RandomForestRegressor(max_depth = 100,random_state=0,n_estimators=5,
                               )


# In[348]:


rf_regr_model = rf_regr.fit(x_train, y_train)


# In[349]:


rf_regr_model


# In[350]:


train_pr = rf_regr_model.predict(x_train)


# In[351]:


fare_preds = rf_regr_model.predict(x_test)


# In[352]:


fare_preds


# In[353]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, fare_preds)


# In[355]:


sqrt(mean_squared_error(y_train, train_pr))


# In[356]:


#RMSE
from math import sqrt
sqrt(mean_squared_error(y_test, fare_preds))


# In[320]:


from sklearn.metrics import r2_score


# In[321]:


r2_score(y_train, train_pr)


# In[339]:


r2_score(y_test, fare_preds)


# In[323]:


rfc_param_grid = {
    'bootstrap': [True],
    'max_depth': [2, 5, 30, 80, 90, 100
                 ],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10, 22],
    #'criterion' : ['gini', 'entropy']
    #,min_impurity_decrease : [???]
    #,min_impurity_split : [???]
    #,class_weight : [???]
}


# In[324]:


from sklearn.model_selection import GridSearchCV
rfc_gs = GridSearchCV(estimator=rf_regr, param_grid=rfc_param_grid, cv= 5,
                     #scoring methods also include: accuracy, balanced accuracy, average precision, f1, etc.  see documentation
                     )


# In[326]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=1)


# In[327]:


dfs = [x_train, x_val]
dfsy = [y_train, y_val]
x_train_rf = pd.concat(dfs)
y_train_rf = pd.concat(dfsy)


# In[328]:


rfc_gs.fit(x_train_rf, y_train_rf)


# In[329]:


rfc_gs.best_params_


# In[330]:


train_pr_grid = rfc_gs.predict(x_train)


# In[331]:


fare_preds_grid = rfc_gs.predict(x_test)


# In[336]:


mean_squared_error(y_train, train_pr_grid)


# In[337]:


sqrt(mean_squared_error(y_train, train_pr_grid))


# In[332]:


mean_squared_error(y_test, fare_preds_grid)


# In[333]:


sqrt(mean_squared_error(y_test, fare_preds_grid))


# In[334]:


r2_score(y_train, train_pr_grid)


# In[338]:


r2_score(y_test, fare_preds_grid)

