import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import category_encoders as ce
from sklearn.inspection import permutation_importance


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 5000)


train_data = pd.read_csv('./train.csv', sep=',', header=0)
weather_data = pd.read_csv('./weather.csv', sep=',', header=0)
print(train_data[1300:1400])

num_of_moskits = train_data.groupby(by=['Date','Address', 'Species','Trap', 'AddressAccuracy', 'Block',
                                        'Street', 'AddressNumberAndStreet', 'Latitude', 'Longitude', 'WnvPresent'])['NumMosquitos'].sum() #делаем чтобы количество москитов отображалось больше 50

train_data = pd.merge(left=train_data, right=num_of_moskits,
                      left_on=['Date', 'Address', 'Species', 'Trap', 'AddressAccuracy', 'Block',
                               'Street', 'AddressNumberAndStreet', 'Latitude', 'Longitude', 'WnvPresent'],
                      right_on=['Date', 'Address', 'Species', 'Trap', 'AddressAccuracy', 'Block',
                                'Street', 'AddressNumberAndStreet', 'Latitude', 'Longitude', 'WnvPresent'],
                      how='inner')
train_data.drop(['NumMosquitos_x'], axis=1, inplace=True)
train_data.rename(columns={'NumMosquitos_y': 'NumMosquitos'}, inplace=True)
print(train_data[1300:1400])
train_data.drop_duplicates(ignore_index=True, inplace=True)



train_data.drop(['Address', 'AddressNumberAndStreet'], axis=1, inplace=True)#убираем колонки которые не несут дополнительной информации


weather_data = weather_data[weather_data['Station'] == 1] #берем показание 1ой станции, так как в показаниях 2ой очень много пропусков
weather_data.drop(['Water1', 'SnowFall', 'CodeSum'], axis=1, inplace=True)



merged_data = pd.merge(train_data, weather_data, on=['Date'], how='left')

merged_data.to_csv('merged_data', sep=',')

