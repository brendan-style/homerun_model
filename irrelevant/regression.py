# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:53:54 2025

@author: Brendan
"""
import pandas as pd
old_hits = pd.read_csv('final_rates_b.csv')
names = old_hits[['player_name','pitcher']]
names = names.drop_duplicates(subset=['player_name','pitcher'], keep='first').reset_index(drop=True)
weighted = pd.DataFrame()
for i in range(0,len(names)):
    name = names['pitcher'][i]
    player = old_hits.query('pitcher == @name').reset_index().drop(columns='index')
    for q in range(0,len(player)):
        for p in range(0,11):
            date = player.year[q]
            player.iloc[:,p][q] = round(player.iloc[:,p][q]*(player['count'][q]/sum(player.query('year == @date')['count'])),5)
    player = player.groupby(['player_name','year','pitch','pitcher']).agg({
        **{col: 'sum' for col in list(player.columns[:11])},
        **{col: 'sum' for col in [player.columns[15]]}}).reset_index()
    if len(player) >= 2:
        player = player.groupby(['player_name','pitcher','year']).agg({
            **{col: 'sum' for col in list(player.columns[4:])},}).reset_index()
    else:
        player.iloc[:,4:15] = (player.iloc[:,4:15]+1)/2
    player = round(player,2)
    player = player.query('count >= 300')
    weighted = weighted.append(player)
weighted = weighted.reset_index(drop=True)
weighted = weighted.drop_duplicates().reset_index(drop=True)
weighted = weighted.query('year != 2024')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr as r2
weighted['gb/fb'] = round(weighted.gb/weighted.fb,2)
X = weighted.iloc[:,3:14].drop(columns=['hr','gb','fb'])
X['gb/fb'] = weighted['gb/fb']
y = weighted.hr
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)
result = LassoCV(cv=5, random_state=79, max_iter=10000)
result = result.fit(X_train, y_train)
coef_df = pd.DataFrame({'Variable': X_train.columns,'Coefficient': result.coef_})


X_test['pred_hr'] = result.predict(X_test).round(2)
X_train['pred_hr'] = result.predict(X_train).round(2)

mse(X_test.pred_hr,y_test)
mse(X_train.pred_hr,y_train)

weighted['pred_hr'] = result.predict(weighted.drop(columns=['hr','gb','fb','count','pred_hr']).iloc[:,3:]).round(2)

#%%