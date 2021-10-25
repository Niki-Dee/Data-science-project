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


test_data = pd.read_csv('./test_truncated.csv', sep=';', header=0)
weather_data = pd.read_csv('./weather.csv', sep=',', header=0)




test_data.drop(['Address', 'AddressNumberAndStreet'], axis=1, inplace=True)#убираем колонки которые не несут дополнительной информации


weather_data = weather_data[weather_data['Station'] == 1] #берем показание 1ой станции, так как в показаниях 2ой очень много пропусков
weather_data.drop(['Water1', 'SnowFall', 'CodeSum'], axis=1, inplace=True)



merged_data = pd.merge(test_data, weather_data, on=['Date'], how='left')
print(merged_data[:10])

merged_data.to_csv('new_test_data', sep=',')

