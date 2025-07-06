from qpython import qconnection
import pykx
from mlbench import MLBench
import joblib
import pandas as pd


# use prepare_ml_daily_data_home.q to prepare the data in the kdb portal 21999
kdbhost = 'localhost'
q = qconnection.QConnection(kdbhost, port=121, username='', password='', pandas=True)
q.open()
qkx = pykx.QConnection(kdbhost, port=121, username='', password='')
data = qkx('`time`sym xasc select from d where not null rtn5, not null rtn5avg, ipodays>30').pd()


result_all = pd.DataFrame()

features_test = [['vola90','macd','bias30','bias30_lag5','bias30_avg5','bias30_avg30','bias100','bias5D30D','idxBias30','idxBias30_lag1','idxBias30_lag5','idxBias30_lag10', 'idxBias30_avg5','idxBias30_avg30','idxBias100','allBias30']]
for rtn_label in ['rtn1','rtn3','rtn5','rtn5avg','rtn10avg']:
    print(f'rtn:{rtn_label}')
    for feature in features_test:
        for pct in [0.6, 0.7, 0.8, 0.9]:
            print(f'features: {feature}')
            data['label'] = data[rtn_label]
            #data['label'] = [1 if x>0 else 0 for x in data[rtn_label]]
            mybench = MLBench(data)
            mybench.select_features(feature)
            mybench.select_model(problem_type='regression', models=['RandomForest','LightGBM'])
            #mlbench.select_model(problem_type='classification', models=['RandomForest','SVM','LightGBM','LogisticRegression','AdaBoost','KNN','XGBoost','CatBoost','NeuralNetwork'])
            result = mybench.test_models(pct)
            
            
            
            joblib.dump(mybench.using_models.get('RandomForest'), './model/RF_'+rtn_label+ '_pct'+ str(pct)+'.joblib')
            print("Random Forest model saved.")
            joblib.dump(mybench.using_models.get('LightGBM'), './model/LGBM_'+rtn_label+ '_pct'+ str(pct)+'.joblib')
            print("LightGBM model saved.")
            
            result_regression = pd.DataFrame.from_dict(result).T
            result_regression['label_name'] = rtn_label
            result_regression['features'] = "|".join(feature)
            result_regression['train_pct'] = pct
            result_all = pd.concat([result_all, result_regression])

result_all.to_csv('./resulta_all.csv')


result_all.groupby('label_name').mean()

result_all.select_dtypes(include=['float64', 'int64']).groupby(level=0).mean()
result_all.select_dtypes(include=['float64', 'int64'])

result_all.drop(['features'], axis=1).groupby('label_name').mean()
result_all.drop(['features','label_name'], axis=1).groupby('train_pct').mean()






#verify the loaded result
bestfeature = ['vola90','macd','bias30','bias30_lag5','bias30_avg5','bias30_avg30','bias100','bias5D30D','idxBias30','idxBias30_lag1','idxBias30_lag5','idxBias30_lag10', 'idxBias30_avg5','idxBias30_avg30','idxBias100','allBias30']
data['label'] = data['rtn5']
benchtest = MLBench(data)
benchtest.select_features(bestfeature)
benchtest.select_model(problem_type='regression', models=['RandomForest','LightGBM'])
result = benchtest.test_models(0.6)

pd.DataFrame.from_dict(result).T


modelmap = {
 'RF_rtn1_pct60': 'RF_rtn1_pct0.6.joblib',
 'RF_rtn3_pct60': 'RF_rtn3_pct0.6.joblib',
 'RF_rtn5_pct60': 'RF_rtn5_pct0.6.joblib',
 'RF_rtn1_pct90': 'RF_rtn1_pct0.9.joblib',
 'RF_rtn3_pct90': 'RF_rtn3_pct0.9.joblib',
 'RF_rtn5_pct90': 'RF_rtn5_pct0.9.joblib',
}



#load saved models
models = {}
for k,v in modelmap.items():
    print(f'{k}:{v}')
    models[k] = joblib.load('./model/'+v)

# make prediction
testdata = data[data['date']>='2025-06-30']
X_test = testdata.loc[:,bestfeature].values
for k,m in models.items():
    print(f'{k}:{v}')
    testdata[k] = m.predict(X_test)

