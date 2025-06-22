
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from qpython import qconnection
import math
import re
import pykx
import seaborn as sns
import warnings
from sklearn.model_selection import  train_test_split
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)


def ml_metrics(ypred,y):
    ypred = np.array(ypred)
    y = np.array(y)
    es = (ypred-y)**2
    mse = np.mean(es)
    mae = np.mean(np.abs(ypred-y))
    R2 = 1- np.sum(es)/np.sum((ypred-np.mean(ypred))**2)
    return pd.DataFrame({'mse':mse, 'mae':mae, 'R2':R2}.items(), columns=['metrics', 'value'])


kdbhost = 'localhost'
q = qconnection.QConnection(kdbhost, port=21978, username='', pandas=True)
q.open()
qkx = pykx.QConnection(kdbhost, port=21978, username='', password='')


data = qkx('`time`sym xasc select from d where date>=2021.01.01, not null rtn5, ipodays>30').pd()

databack = data


data.groupby(['sym']).agg({'sym':'count','close':'mean','notional':'sum','bias30':'mean','rtn5':'mean'})

data['sym'].unique()


len(data)

print(f'rawdata  shape:{data.shape}')
data.describe()


#if want to normalize the value, use below
column_need_norm = []
#filter
for x in data.columns.values:
    if any(s in x.lower() for s in ['bias','speed']) and not any(s in x.lower() for s in ['idx','all','rank']):
        column_need_norm.append(x)


data[column_need_norm] = data[column_need_norm].div(data['vola90'], axis=0)
column_need_norm


def func_label(x, s):
    if x > s:
        return 1
    elif x < -s:
        return 0
    else:
        return 0.5

def func_label_multiclass(x, s):
    if x > s:
        return 2
    elif x < -s:
        return 0
    else:
        return 1
    
    
def sigmoid_label(x, lamda):
    return 1./(1+math.exp(lamda*x))

#data['label'] = [func_label(x, 50) for x in data['bps']]
#data['label'] = data['bps'].rank(pct=True)
#data['label'] = [func_label_multiclass(x, 110) for x in data['rtn1d']]
#data['label'] = [func_label(x, 220) for x in data['rtn1d']]
#data['label'] = [func_label(x, 300) for x in data['rtn1d']]
#data['label'] = [func_label(x, 100) for x in data['rtn30']]
#data['label'] = [func_label(x, 150) for x in data['rtn30']]
#data.groupby(data['label'].apply(pd.qcut(data['rtn1d'],10)).agg({'label': ['count','mean'],'rtn1':['mean'],'rtn30':['mean'],'rtn8H':['mean'], 'rtn1d':['mean']})
#data.groupby(['label']).agg({'label': ['count','mean'], 'rtn1':['mean'], 'rtn5':['mean'], 'rtn5avg':['mean'], 'rtn10avg':['mean']})
#data.groupby(pd.qcut(data['rtn1'],10)).agg({'label': ['count','mean'], 'rtn1':['mean'], 'rtn5':['mean'], 'rtn5avg':['mean'], 'rtn10avg':['mean']})


xcol = []

#filter
for x in data.columns.values:
    if any(s in x.lower() for s in ['vola','bias','macd','rsi','rank']):
        xcol.append(x)
xcol
import matplotlib.pyplot as plt

#print(data.loc[:,xcol].corr())
plt.matshow(data.loc[:,xcol].corr())
plt.show()

xcol
#xcol = ['vola90','macd','rsi24','bias30','bias100','idxBias30','rankNtl100', 'rankVolRatio5D30D']
#xcol = ['vola90','macd','bias30','idxBias30','allBias30','rankNtl100']
xcol = ['vola90','macd','bias30','bias30_lag5','bias30_avg5','bias30_avg30','bias100','bias5D30D','idxBias30','idxBias30_lag1','idxBias30_lag5','idxBias30_lag10', 'idxBias30_avg5','idxBias30_avg30','idxBias100','allBias30']
X = data.loc[:,xcol].values
print(xcol)

data['label'] = data['rtn5']
#data['label'] = (data['rtn1']+data['rtn3']+data['rtn5']+data['rtn10'])/4
#data['label'] = data['rtn5']/data['vola90']
y = data.loc[:,['label']].values


#another way of splitting, but payattention, below method is biased, in sample used
# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


nsplit = int(0.6*len(data))
print(data.iloc[nsplit])
X_train, X_test = X[:nsplit,] , X[nsplit:,]
y_train, y_test = y[:nsplit,] , y[nsplit:,]
#scaler = StandardScaler()
scaler = RobustScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

X_train


# create dataset for lightgbm
# if you want to re-use data, remember to set free_raw_data=False
train_data = lgb.Dataset( X_train, y_train, feature_name=xcol)
#lgb_eval = lgb.Dataset(X_test, y_test, reference=train_data)

params = {
    'objective': 'regression',  # regression问题
    #'objective': 'multiclass',  # multiclass
    'metric': 'rmse', # root square loss, aliases: root_mean_squared_error, l2_root
    #'metric': 'l1',  #absolute loss, aliases: mean_absolute_error, mae, regression_l1
    #'metric': 'multi_logloss',  # 使用log损失作为评估指标
    #'num_class': len(np.unique(y_train)),
    #'boosting_type': 'gbdt',  # 使用 GBDT 算法
    #'num_leaves': 31,  # 叶子节点数
    #'learning_rate': 0.05,  # 学习率
    #'feature_fraction': 0.9,  # 特征采样比例
    #'bagging_fraction': 0.8,  # 数据采样比例
    #'bagging_freq': 5,  # 每 5 次迭代进行一次 bagging
    #'verbose': 0  # 不输出详细信息
}


print("Starting training...")
# feature_name and categorical_feature
model = lgb.train(params, train_data, num_boost_round=100)
#model = lgb.train( params,  train_data,   num_boost_round=10,   valid_sets=lgb_train  # eval training data )
y_testpred = model.predict(X_train)
y_pred = model.predict(X_test)



# use another model, random forest
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
y_testpred = model.predict(X_train)
y_pred = model.predict(X_test)


# 预测

#y_pred_binary = np.round(y_pred)  # 将概率值转换为二进制分类



pd.Series(y_pred.flatten()).quantile(np.linspace(0, 1, 10, 0), 'lower')


pd.Series(y_test.flatten()).quantile(np.linspace(0, 1, 10, 0), 'lower')



data_insample = data.iloc[0:nsplit,]
data_insample.loc[:,'pred'] = y_testpred
data_test = data.iloc[nsplit:,]
data_test.loc[:,'pred'] = y_pred
data1 = data_test
#data1 = data_insample
print(data1.shape)
#data1['pred_rank'] =  data1['pred'].rank(pct=True)
#result = data1.groupby (['pred']) .agg(
#result = data1.groupby (data1['pred_rank'].apply(lambda x: round(x,1))) .agg(
result = data1.groupby (pd.qcut(data1['pred'], 2) ) .agg(
#result = data1.groupby (data_test['pred'].apply(lambda x:x>=0.8)) .agg(
#result = data1.groupby (['date']) .agg(
 n=('sym', 'count'), 
 nsym = ('sym','nunique'),
 nday = ('date','nunique'),
 time = ('time', 'nunique'), 
 macd=('macd', 'mean'), 
 rsi=('rsi24', 'mean'),    
 bias30=('bias30', 'mean'),
 bias100=('bias100','mean'), 
 pred=('pred', 'mean'), 
 rtn1=('rtn1', 'mean'),
 rtn3=('rtn3', 'mean'),
 rtn5=('rtn5', 'mean'),
 rtn10=('rtn10', 'mean'),
 rtn30=('rtn30', 'mean'),   
)

#result = result.reset_index()
result


pred_res = pd.DataFrame({'pred': np.append(y_testpred,y_pred), 'flag': np.append(len(y_testpred)*[0],len(y_pred)*[1])   })
#q.sendSync( '{pred::x}', pred_res)
q.sendSync( '{pred::x}', data1)


resultback = result


y_pred



feature_importance = pd.DataFrame({'feature_name': model.feature_name(), 'importance': model.feature_importance()})
feature_importance = feature_importance.sort_values(['importance'], ascending=[True])
#plt.plot(feature_importance['feature_name'],feature_importance['importance'])
feature_importance.plot.barh(x='feature_name', y='importance',fontsize=6)



feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':xcol})
#plt.figure(figsize=100)
sns.set(font_scale =0.5)
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:40])
#sns.barplot(x=feature_imp, y=feature_imp.index)

#for random forest
feature_imp = pd.Series(model.feature_importances_, index = xcol ).sort_values(ascending=False)
#feature_imp[feature_imp>0.03]

sns.barplot(x=feature_imp[0:30], y=feature_imp.index[0:30])


def holding_func(x, s):
    if x > s:
        return 1
    elif x < -s:
        return -1
    else:
        return 0



def init_strategy(datatest, invest_each, hold_threshold):   
    invest_each = 1/len(datatest['sym'].unique())
    print(f'invest_each:{invest_each}')
    datatest['side'] = [holding_func(x, hold_threshold) for x in datatest['pred']]
    datatest['bps'] = datatest['side']*datatest['rtn3']
    datatest['pnl'] = invest_each*datatest['bps']/10000
    #data1.groupby (['side']).agg( n=('sym', 'count'), )
    data1idx = datatest[datatest['sym'] == 'BTC/USDT.BN'][['time','close']]
    data1idx = data1idx.rename(columns={'close': 'closeIdx'})
    datatest = datatest.merge(data1idx, on='time', how='inner')
    return(datatest)


def cal_strategy_matrics(datatest):
        pnl = datatest.groupby (['date']).agg(
         n=('bias30', 'count'), 
         nsym = ('sym','nunique'),
         time = ('time', 'nunique'), 
          pred=('pred', 'mean'), 
          rtn1=('rtn1', 'mean'),
          rtn5=('rtn5', 'mean'),
          bps=('bps', 'mean'),
          pnl=('pnl', 'sum'),
          side=('side', 'sum'),  
          closeIdx=('closeIdx', 'mean'),  
        )
        pnl['pnlsums'] = pnl['pnl'].cumsum()



data1 = data_test
invest_each = 1/len(data1['sym'].unique())
print(f'invest_each:{invest_each}')
data1['side'] = [holding_func(x, 0) for x in data1['pred']]
data1['bps'] = data1['side']*data1['rtn3']
data1['pnl'] = invest_each*data1['bps']/10000
#data1.groupby (['side']).agg( n=('sym', 'count'), )
data1idx = data1[data1['sym'] == 'BTC/USDT.BN'][['time','close']]
data1idx = data1idx.rename(columns={'close': 'closeIdx'})
data1 = data1.merge(data1idx, on='time', how='inner')

#eval the model using real pnl
pnl = data1.groupby (['date']).agg(
#pnl = data1[data1['pred']<0.0].groupby (['date']).agg(
 n=('bias30', 'count'), 
 nsym = ('sym','nunique'),
 time = ('time', 'nunique'), 
  pred=('pred', 'mean'), 
  rtn1=('rtn1', 'mean'),
  rtn5=('rtn5', 'mean'),
  bps=('bps', 'mean'),
  pnl=('pnl', 'sum'),
  side=('side', 'sum'),  
  closeIdx=('closeIdx', 'mean'),  
)

pnl['pnlsums'] = pnl['pnl'].cumsum()
#pnl['pnlsums'].plot()

plt.plot(pnl['pnlsums'], label='pnlsums')
plt.plot(pnl['side'], label='side')



fig, ax1 = plt.subplots()
ax1.plot(pnl['pnlsums'], 'b-', label='pnlsums')
#ax1.set_ylabel('pnlsums', color='b')
ax2 = ax1.twinx()
#ax2.plot(pnl['side'], 'r-', label='side')
ax2.plot(pnl['closeIdx'], 'r-', label='closeIdx')

#ax2.set_ylabel('side', color='r')
#plt.title('pnlsum & side')
fig.legend()
plt.show()





data['month'] = data['date'].dt.to_period('M').dt.to_timestamp('D')

datelist = data['month'].unique()

#xcol = ['vola90','macd','rsi24','bias30','bias100','idxBias30','rankNtl30', 'rankVolRatio5D30D']
#xcol = ['vola90','macd','bias30','bias100','bias5D30D','idxBias30','idxBias100','allBias30']
xcol = ['vola90','macd','bias30','bias30_lag5','bias30_avg5','bias30_avg30','bias100','bias5D30D','idxBias30','idxBias30_lag1','idxBias30_lag5','idxBias30_lag10', 'idxBias30_avg5','idxBias30_avg30','idxBias100','allBias30']
data['label'] = data['rtn5']
#data['label'] = data['rtn5']/data['vola90']

#look each month and recalibrate using the acumulated data
pred_res = pd.DataFrame()
for i in range(20,len(datelist)):
#for i in range(20,22):
    print(datelist[i-1], datelist[i])
    data_train = data[data['month']<=datelist[i-1]]
    #data_train = data_train[data_train['month']>=datelist[i-19]]
    X_train = data_train.loc[:,xcol].values
    y_train = data_train.loc[:,['label']].values
    print(f'len of train:{len(X_train)}')
    
    data_test = data[data['month']==datelist[i]]
    X_test = data_test.loc[:,xcol].values
    y_test = data_test.loc[:,['label']].values
    print(f'len of test:{len(X_test)}')
    
    
    lgb_data = lgb.Dataset( X_train, y_train, feature_name=xcol)
    model = lgb.train(params, lgb_data, num_boost_round=100)
    
    #model = RandomForestRegressor(n_estimators=100, random_state=0)
    #model.fit(X_train, y_train)
    
    data_test.loc[:,'pred'] = model.predict(X_test)
    pred_res = pd.concat([pred_res, data_test])



invest_each = 1/ len(pred_res['sym'].unique())
pred_res['side'] = [holding_func(x, 0) for x in pred_res['pred']]
pred_res['bps'] = pred_res['side']*pred_res['rtn1']
#pred_res['invest'] = [ invest_each*max(1,abs(x)/300) for x in pred_res['bps'] ]
#pred_res['pnl'] = pred_res['invest']*pred_res['bps']/10000
pred_res['pnl'] = invest_each*pred_res['bps']/10000
#eval the model using real pnl
pnl = pred_res.groupby (['date']).agg(
#pnl = data1[data1['pred']<0.0].groupby (['date']).agg(
 n=('bias30', 'count'), 
 nsym = ('sym','nunique'),
 time = ('time', 'nunique'), 
  pred=('pred', 'mean'), 
  rtn1=('rtn1', 'mean'),
  rtn5=('rtn5', 'mean'),
  bps=('bps', 'mean'),
  pnl=('pnl', 'sum'),
)

pnl['pnlsums'] = pnl['pnl'].cumsum()
pnl['pnlsums'].plot()


