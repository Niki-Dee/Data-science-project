import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import category_encoders as ce
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#Настройки отображения таблицы
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 5000)


train_data = pd.read_csv('./merged_data', sep=',', header=0)


train_data['Date'] = train_data['Date'].apply(lambda x: x.split('-')[0] + x.split('-')[1] + x.split('-')[2]) #переводим дату в число
train_data['Date'].astype(int)


t_data = np.array(train_data['WetBulb'].values[train_data['WetBulb'] != 'M'])
t_data = t_data.astype(float)
train_data['WetBulb'] = np.where(train_data['WetBulb'] == 'M', np.mean(t_data), train_data['WetBulb']) #замена на среднее значение
train_data['PrecipTotal'] = np.where(train_data['PrecipTotal']=='  T', 0.001, train_data['PrecipTotal'])



t_data = np.array(train_data['StnPressure'].values[train_data['StnPressure'] != 'M'])
t_data = t_data.astype(float)
train_data['StnPressure'] = np.where(train_data['StnPressure'] == 'M', np.round(np.mean(t_data), 2), train_data['StnPressure'])


encoder_street = ce.TargetEncoder(min_samples_leaf=10, smoothing=40).fit(train_data['Street'], train_data['AddressAccuracy'])
encoder_species = ce.TargetEncoder(min_samples_leaf=10, smoothing=40).fit(train_data['Species'], train_data['AddressAccuracy'])


train_data['Species'] = encoder_species.transform(train_data['Species']) # енкодим строковые даные
train_data['Street'] = encoder_street.transform(train_data['Street'])

def clear_trap(x): #убираем опечатки
    x = x.split('T')[1]
    if x[-1] == 'B':
        x = x.replace('B', '')
    elif x[-1] == 'C':
        x = x.replace('C', '')
    return x
train_data['Trap'] = train_data['Trap'].apply(clear_trap)


x = train_data[['Date', 'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'Tmax', 'Tmin',
          'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure', 'SeaLevel',
          'ResultSpeed', 'ResultDir', 'AvgSpeed'] ].copy()
y = train_data['NumMosquitos']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42) #разбиваем на тестовую и обучающуюся группы


mdl = LinearRegression(fit_intercept=True)
mdl.fit(x_train,y_train)
mdl_predict = mdl.predict(x_test)

#Проверка важности модели
# importances = permutation_importance(
#     mdl, x_test, y_test,
#     n_repeats=10, random_state=42
# )
# for i, column in enumerate(x_test.columns):
#     print(column, importances.importances_mean[i], importances.importances_std[i])
# print(mean_squared_error(y_test,mdl_predict))
#
x_train.drop([ 'ResultDir', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
               'Date', 'StnPressure', 'SeaLevel'], axis=1, inplace=True)
x_test.drop([ 'ResultDir', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
              'Date', 'StnPressure', 'SeaLevel'], axis=1, inplace=True)
#
mdl.fit(x_train,y_train)
mdl_predict = mdl.predict(x_test)

print(mean_squared_error(y_test, mdl_predict))



x_test['LowSunrise'] = np.where(x_test['Sunrise'] <= 472, x_test['Sunrise'], 0) #по графику видим что нужно разбить переменные на 2, для лучшего обучения моделей
x_test['HighSunrise'] = np.where(x_test['Sunrise'] >= 486, x_test['Sunrise'], 0)
x_test.drop(['Sunrise'], axis=1, inplace=True)
x_train['LowSunrise'] = np.where(x_train['Sunrise'] <= 472, x_train['Sunrise'], 0) #по графику видим что нужно разбить переменные на 2, для лучшего обучения моделей
x_train['HighSunrise'] = np.where(x_train['Sunrise'] >= 486, x_train['Sunrise'], 0)
x_train.drop(['Sunrise'], axis=1, inplace=True)


x_test['LowLatitude'] = np.where(x_test['Latitude'] <= 41.9, x_test['Latitude'], 0)
x_test['HighLatitude'] = np.where(x_test['Latitude'] > 41.9, x_test['Latitude'], 0)
x_test.drop(['Latitude'], axis=1, inplace=True)
x_train['LowLatitude'] = np.where(x_train['Latitude'] <= 41.9, x_train['Latitude'], 0)
x_train['HighLatitude'] = np.where(x_train['Latitude'] > 41.9, x_train['Latitude'], 0)
x_train.drop(['Latitude'], axis=1, inplace=True)

# print(x_test[1:5])
mdl.fit(x_train,y_train)
mdl_predict = mdl.predict(x_test)

print(mean_squared_error(y_test, mdl_predict))

x_test['Tmax'] = np.log1p(x_test['Tmax'])
x_train['Tmax'] = np.log1p(x_train['Tmax'])
x_test['Tmax'] = np.log1p(x_test['Tmax'])
x_train['Tmax'] = np.log1p(x_train['Tmax'])

x_test['Block'] = np.log1p(x_test['Block'])
x_train['Block'] = np.log1p(x_train['Block'])
x_test['Block'] = np.log1p(x_test['Block'])
x_train['Block'] = np.log1p(x_train['Block'])


mdl.fit(x_train,y_train)
mdl_predict = mdl.predict(x_test)

print(mean_squared_error(y_test, mdl_predict))


x_test['LowSunrise'] = np.log1p(x_test['LowSunrise'])
x_train['LowSunrise'] = np.log1p(x_train['LowSunrise'])


x_test['HighLatitude'] = np.log1p(x_test['HighLatitude'])
x_train['HighLatitude'] = np.log1p(x_train['HighLatitude'])



# mdl.fit(x_train,y_train)
# mdl_predict = mdl.predict(x_test)
#
# print(mean_squared_error(y_test, mdl_predict))

x_train['Trap'] = pd.to_numeric(x_train['Trap'])
x_test['Trap'] = pd.to_numeric(x_test['Trap'])
dtrain = xgb.DMatrix(x_train, y_train, )
dtest = xgb.DMatrix(x_test, y_test, )

params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'subsample': 1,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.3,
        'gamma': 1.,
        'max_depth': 3,
        'min_child_weight': 8,
        'seed': 32,}
xgb_model = xgb.XGBRegressor(**params)
# подбор параметров
# params_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.5],
#                }
# clf = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=params_grid,
#     scoring='neg_mean_squared_error',
#     cv=4
# #6016
# )
# clf.fit(x_train, y_train)
# grid = pd.DataFrame(clf.cv_results_)
# print(grid)
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=50,
    early_stopping_rounds=20,
    evals=[(dtrain, 'Train'), (dtest, 'Test')]
)
prediction = model.predict(dtest)
print(np.sqrt(mean_squared_error(y_test, prediction))) # показатель не улучшился

# tree1 = DecisionTreeRegressor(criterion='squared_error',  max_depth=9,  min_samples_split=5,  min_samples_leaf=5).fit(x_train, y_train)
# prediction = tree1.predict(x_test)
# print(mean_squared_error(y_test, prediction))

#построение графиков для просмотра распределения переменных
# plt.subplot(3,4,1)
# plt.hist(x= x['Species'])
# plt.title('Species')
# plt.subplot(4,3,2)
# plt.hist(x= x['Block'])
# plt.title('Block')
# plt.subplot(4,3,3)
# plt.hist(x= x['Trap'])
# plt.title('Trap')
# plt.subplot(4,3,4)
# plt.subplot(4,3,5)
# plt.hist(x= x['Longitude'])
# plt.title('Longitude')
# plt.subplot(4,3,6)
# plt.hist(x= x['Tmax'])
# plt.title('Tmax')
# plt.subplot(4,3,7)
# plt.hist(x= x['Tmin'])
# plt.title('Tmin')
# plt.subplot(4,3,8)
# plt.hist(x= x['Tavg'])
# plt.title('Tavg')
# plt.subplot(4,3,9)
# plt.subplot(4,3,10)
# plt.subplot(4,3,11)
# plt.hist(x= x['SeaLevel'])
# plt.title('SeaLevel')

#Подготовка тестовой таблицы
test_data = pd.read_csv('./new_test_data', sep=',', header=0)


t_data = np.array(test_data['WetBulb'].values[test_data['WetBulb'] != 'M'])
t_data = t_data.astype(float)
test_data['WetBulb'] = np.where(test_data['WetBulb'] == 'M', np.mean(t_data), test_data['WetBulb']) #замена на среднее значение




t_data = np.array(test_data['StnPressure'].values[test_data['StnPressure'] != 'M'])
t_data = t_data.astype(float)
test_data['StnPressure'] = np.where(test_data['StnPressure'] == 'M', np.round(np.mean(t_data), 2), test_data['StnPressure'])


encoder_street = ce.TargetEncoder(min_samples_leaf=10, smoothing=40).fit(test_data['Street'], test_data['AddressAccuracy'])
encoder_species = ce.TargetEncoder(min_samples_leaf=10, smoothing=40).fit(test_data['Species'], test_data['AddressAccuracy'])


test_data['Species'] = encoder_species.transform(test_data['Species']) # енкодим строковые даные
test_data['Street'] = encoder_street.transform(test_data['Street'])

test_data = test_data[['Date', 'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'Tmax', 'Tmin',
          'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure', 'SeaLevel',
          'ResultSpeed', 'ResultDir', 'AvgSpeed']]

final_test_data = test_data[['Date', 'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'Tmax', 'Tmin',
          'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure', 'SeaLevel',
          'ResultSpeed', 'ResultDir', 'AvgSpeed']]

test_data.drop([ 'ResultDir', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
               'Date', 'StnPressure', 'SeaLevel'], axis=1, inplace=True)


test_data['LowSunrise'] = np.where(test_data['Sunrise'] <= 472, test_data['Sunrise'], 0) #по графику видим что нужно разбить переменные на 2, для лучшего обучения моделей
test_data['HighSunrise'] = np.where(test_data['Sunrise'] >= 486, test_data['Sunrise'], 0)
test_data.drop(['Sunrise'], axis=1, inplace=True)

test_data['LowLatitude'] = np.where(test_data['Latitude'] <= 41.9, test_data['Latitude'], 0)
test_data['HighLatitude'] = np.where(test_data['Latitude'] > 41.9, test_data['Latitude'], 0)
test_data.drop(['Latitude'], axis=1, inplace=True)

test_data['Tmax'] = np.log1p(test_data['Tmax'])

test_data['Block'] = np.log1p(test_data['Block'])

test_data['LowSunrise'] = np.log1p(test_data['LowSunrise'])
test_data['HighLatitude'] = np.log1p(test_data['HighLatitude'])

# #расчитаем количество москитов
# final_test_data['NumMosquitos'] = np.round(np.sqrt(mdl.predict(test_data)), 0)
# #запишем результаты в файл
# final_test_data.to_csv('new_test_data', sep=',')

#начнем расчет вероятности москитов
train_data = pd.read_csv('./merged_data', sep=',', header=0)

train_data['Date'] = train_data['Date'].apply(lambda x: x.split('-')[0] + x.split('-')[1] + x.split('-')[2]) #переводим дату в число
train_data['Date'].astype(int)


t_data = np.array(train_data['WetBulb'].values[train_data['WetBulb'] != 'M'])
t_data = t_data.astype(float)
train_data['WetBulb'] = np.where(train_data['WetBulb'] == 'M', np.mean(t_data), train_data['WetBulb']) #замена на среднее значение
train_data['PrecipTotal'] = np.where(train_data['PrecipTotal']=='  T', 0.001, train_data['PrecipTotal'])



t_data = np.array(train_data['StnPressure'].values[train_data['StnPressure'] != 'M'])
t_data = t_data.astype(float)
train_data['StnPressure'] = np.where(train_data['StnPressure'] == 'M', np.round(np.mean(t_data), 2), train_data['StnPressure'])


encoder_street = ce.TargetEncoder(min_samples_leaf=10, smoothing=40).fit(train_data['Street'], train_data['AddressAccuracy'])
encoder_species = ce.TargetEncoder(min_samples_leaf=10, smoothing=40).fit(train_data['Species'], train_data['AddressAccuracy'])


train_data['Species'] = encoder_species.transform(train_data['Species']) # енкодим строковые даные
train_data['Street'] = encoder_street.transform(train_data['Street'])

train_data['Trap'] = train_data['Trap'].apply(clear_trap)

x = train_data[['Date', 'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'Tmax', 'Tmin',
          'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure', 'SeaLevel',
          'ResultSpeed', 'ResultDir', 'AvgSpeed', 'NumMosquitos']].copy()
y = train_data['WnvPresent'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #разбиваем на тестовую и обучающуюся группы

x_train.drop(['Cool', 'DewPoint', 'Tavg', 'AddressAccuracy',
               'Date',  'Tmin', 'Block', 'SeaLevel', 'Heat', 'Sunrise', 'StnPressure',
              'Longitude', 'Latitude', 'Street','ResultSpeed', 'Trap',  'ResultDir', 'Species'], axis=1, inplace=True)
x_test.drop([ 'Cool', 'DewPoint', 'Tavg', 'AddressAccuracy',
              'Date',  'Tmin','Block','SeaLevel', 'Heat', 'Sunrise', 'StnPressure', 'Longitude', 'Latitude',
              'Street', 'ResultSpeed', 'Trap', 'ResultDir', 'Species'], axis=1, inplace=True)

# mdl = LogisticRegression(fit_intercept=True)
# mdl.fit(x_train, y_train)
# mdl_predict = mdl.predict(x_test)
#
# importances = permutation_importance(
#     mdl, x_test, y_test,
#     n_repeats=10, random_state=42
# )
# for i, column in enumerate(x_test.columns):
#     print(column, importances.importances_mean[i], importances.importances_std[i])
# print(roc_auc_score(y_test, mdl_predict))

#Производим крос валидацию
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#
# def estimate_cv(reg_param):
#     test_res = []
#     for train_idx, test_idx in kf.split(x, y):
#         X_train, y_train = x.loc[train_idx].copy(), y[train_idx]
#         X_test, y_test = x.loc[test_idx].copy(), y[test_idx]
#         mdl = LogisticRegression(penalty='l1',  C=1 / reg_param,  solver='liblinear', )
#         mdl = mdl.fit(X_train, y_train)
#         y_test_preds = mdl.predict_proba(X_test)[:, 1]
#         test_res.append(np.round(roc_auc_score(y_test, y_test_preds), 4))
#     return np.round(np.mean(test_res), 4), np.round(np.std(test_res), 4)
#
#
# for reg_param in [2.1, 2.3, 2.7, 3]:
#     test_error, test_error_std = estimate_cv(reg_param)
#     print(f'Regularizatoin: {reg_param} -  test error: {test_error} +- {test_error_std}')

# model = LogisticRegression(penalty='l1',C=1/3,solver='liblinear').fit(x_train, y_train)
# prediction = model.predict(x_test)
# print(roc_auc_score(y_test, prediction)) # показник не змінився

# svm = SVC()
# params_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],'C': [0.5, 1, 2, 5]}
# clf = GridSearchCV(estimator=svm,  param_grid=params_grid, scoring='f1_macro',  cv=4  )
# clf = clf.fit(x_train, y_train)
# grid_search_results = pd.DataFrame(clf.cv_results_)
# print(grid_search_results)

# 'C': 5, 'kernel': 'poly' - найкращий показник
# svm = SVC(C=5, kernel='poly').fit(x_train, y_train)
# prediction = svm.predict(x_test)
# print(roc_auc_score(y_test, prediction)) # показник збільшився

# mdl = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=9, min_samples_split=10, min_samples_leaf=5,
#                              max_features=0.8,  bootstrap=True, max_samples=0.8,  random_state=42  )
#
# mdl = mdl.fit(x_train, y_train)
# prediction = mdl.predict(x_test)
# print(roc_auc_score(y_test, prediction)) # показник збільшився
#
params = {'objective': 'binary:logistic', 'learning_rate': 0.1, 'subsample': 1, 'colsample_bytree': 1,
          'colsample_bylevel': 0.4, 'reg_lambda': 0.1, 'gamma': 0.5, 'max_depth': 5, 'min_child_weight': 10,
          'eval_metric': 'auc', 'silent': 1, 'seed': 32, 'n_estimators': 30}
print(x_train.head(5))
x_train['WetBulb'] = pd.to_numeric(x_train['WetBulb'])
xgb_mdl = xgb.XGBClassifier(**params)
# params_grid = {'gamma':[0.1, 0.5, 0.9, 1.5]}
# clf = GridSearchCV(
#     estimator=xgb_mdl,
#     param_grid=params_grid,
#     scoring='roc_auc',
#     cv=4)
#
# clf.fit(x_train, y_train)
# grid = pd.DataFrame(clf.cv_results_)
# print(grid)

test_data = pd.read_csv('./new_test_data', header=0, sep=',')
test_data1 = pd.read_csv('./test_truncated.csv', header=0, sep=';')
test_data.drop(['Cool', 'DewPoint', 'Tavg', 'AddressAccuracy',
               'Date',  'Tmin', 'Block', 'SeaLevel', 'Heat', 'Sunrise', 'StnPressure',
                'Longitude', 'Latitude', 'Street','ResultSpeed', 'Trap',  'ResultDir', 'Species'], axis=1, inplace=True)

# test_data['WnvPresent'] = test_data1['WnvPresent']
print(test_data.head())

x_train['WetBulb']= pd.to_numeric(x_train['WetBulb'])
x_test['WetBulb']= pd.to_numeric(x_test['WetBulb'])

x_predict = test_data[['Tmax',  'Depart','WetBulb' , 'Sunset',  'AvgSpeed',  'NumMosquitos']]


dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)

params = {'objective': 'binary:logistic', 'learning_rate': 0.1, 'subsample': 1, 'colsample_bytree': 1,
          'colsample_bylevel': 0.4, 'reg_lambda': 0.1, 'gamma': 0.5, 'max_depth': 5, 'min_child_weight': 10,
          'eval_metric': 'auc', 'silent': 1, 'seed': 32, 'n_estimators': 30}

mdl = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=50,
    early_stopping_rounds=20,
    evals=[(dtrain, 'Train'), (dtest, 'Test')]
)

dmatrix = xgb.DMatrix(x_predict)
prediction = mdl.predict(dmatrix)

final_data = pd.read_csv('test_truncated.csv', sep=';', header=0)
final_data['WnvPresent'] = prediction
final_data.to_csv('test_truncated.csv', sep=',')


