# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 18:08:50 2025

@author: Brendan
"""
#%% collect pitches
import pybaseball as bb
import pandas as pd
from numpy import nan

hitters = pd.read_csv('all_batters.csv')
pitches_b = pd.read_csv('all_pitches.csv')
hitters = hitters.rename(columns={'last_name, first_name':'name'})
hitters = hitters.drop_duplicates(subset=['name','player_id'], keep='first').reset_index(drop=True)

# NEED to get hit location next time for pulled flyball rate
for i in range(0,len(hitters)):
    stats = bb.statcast_batter("2025-03-01","2025-11-01",592450)#hitters.iloc[:,1][i])
    stats = stats.query('game_type == "R"').dropna(subset='pitch_type')
    stats = stats[~stats['pitch_type'].isin(['SC','PO','CS','FA','EP',nan,'AB','FC','IN'])]
    stats = stats[['game_year','player_name','pitch_type','release_speed','events','description','stand','p_throws',
             'bb_type','zone','launch_speed','spin_axis','launch_angle','release_spin_rate','estimated_woba_using_speedangle','hit_distance_sc',
             'launch_speed_angle','age_bat']]
    stats['playerid'] = hitters.iloc[:,1][i]
    if i == 0:
        pitches_b = stats
    else:
        pitches_b = pitches_b.append(stats)
pitches_b = pitches_b.reset_index().drop(columns='index').drop_duplicates()
pitches_b.to_csv('all_pitches.csv',index=False)

#%%
import pybaseball as bb
import pandas as pd
from numpy import nan

hitters = pd.read_csv('all_pitchers.csv')
pitches_b = pd.read_csv('all_pitches_p.csv')
hitters = hitters.rename(columns={'last_name, first_name':'name'})
hitters = hitters.drop_duplicates(subset=['name','player_id'], keep='first').reset_index(drop=True)

            # i goes here
for i in range(451,len(hitters)):
    stats = bb.statcast_pitcher("2015-03-01","2025-11-01",hitters.iloc[:,1][i])
    stats = stats.query('game_type == "R"').dropna(subset='pitch_type')
    stats = stats[~stats['pitch_type'].isin(['SC','PO','CS','FA','EP',nan,'AB','FC','IN'])]
    stats = stats[['game_year','game_date','player_name','pitch_type','release_speed','events','description','stand','p_throws','release_pos_z','release_pos_x','plate_x','plate_z',
             'bb_type','zone','launch_speed','spin_axis','launch_angle','release_spin_rate','release_extension','estimated_woba_using_speedangle','hit_distance_sc','n_priorpa_thisgame_player_at_bat',
             'launch_speed_angle','attack_angle','attack_direction','swing_path_tilt','age_bat','age_pit','bat_speed','swing_length','at_bat_number','arm_angle','n_thruorder_pitcher']]
    stats['playerid'] = hitters.iloc[:,1][i]
    if i == 0:
        pitches_b = stats
    else:
        pitches_b = pitches_b.append(stats)
pitches_b = pitches_b.reset_index().drop(columns='index').drop_duplicates()
pitches_b.to_csv('all_pitches_p.csv',index=False)
#%% batters
from numpy import select, nan,inf
import pandas as pd
from unidecode import unidecode
from statistics import mean

fbb = batters.query('in_play ==1')
fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])

grouped = batters.groupby(['player_name','playerid', 'game_year', 'pitch_type']).agg(
hh=('hh', 'sum'), # Count of batted balls with exit velocity over 95 mph
barrels =('barrel','sum'),
poorly_hit=('weak','sum'),
fly_ball =('fly_ball','sum'),
ground_ball =('ground_ball','sum'),
line_drive =('line_drive','sum'),
whiff =('whiff','sum'),
chase=('chase','sum'),
swing =('swing','sum'),
home_run =('home_run','sum'),
bip=('in_play','sum'),
pitch_count=('pitch_type', 'size'),
age=('age_bat','mean')
).reset_index()

xgrouped = fbb.groupby(['player_name','playerid', 'game_year', 'pitch_type']).agg(
tot_wob =('estimated_woba_using_speedangle','sum'),
max_ev=('launch_speed','max'),
count=('pitch_type','size')).reset_index()
xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
xgrouped = xgrouped[['player_name','playerid','game_year','pitch_type','xwobacon','max_ev']]
grouped = grouped.merge(xgrouped,on=['player_name','playerid', 'game_year', 'pitch_type'])
grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid', 'game_year', 'pitch_type'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name','playerid', 'game_year', 'pitch_type']).index).values


# getting averages for every pitch

pitch_list = list(grouped.pitch_type.unique())
pitch_avgs = grouped.groupby('pitch_type').agg({**{col: 'sum' for col in list(grouped.columns[4:16])}})
pitch_avgs[['xwobacon','max_ev','ev','la']] = 0
for pitch in pitch_list:
    subset = grouped.query('pitch_type == @pitch').reset_index(drop=True)
    for col in pitch_avgs.columns[12:]:
        total = 0
        for i in range(len(subset)):
            if col == 'max_ev':
                total = mean(subset.max_ev)
            else:
                value = subset[col][i]*(subset.pitch_count[i]/sum(subset.pitch_count)).round(5)
                total += value
        total = round(total,3)
        pitch_avgs[col][pitch] = total
pitch_avgs['gb/fb'] = (pitch_avgs['ground_ball']/pitch_avgs['fly_ball']).round(2)
pitch_avgs = pitch_avgs.drop(columns=['fly_ball','ground_ball'])

for col in pitch_avgs.columns[:8]:
    if col in ['chase','whiff']:
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['swing'])*100,2)
    elif col == 'swing':
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['pitch_count'])*100,2)
    elif col=='home_run':
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['bip'])*100,4)
    else:
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['bip'])*100,2)
pitch_avgs = pitch_avgs.reset_index()


# getting league averages to regress small sample sizes

league_sums= pd.DataFrame((grouped.groupby('game_year').agg({
    **{col: 'sum' for col in list(grouped.columns[4:16])}})).sum().reset_index()).T
league_sums.columns = league_sums.loc['index']
league_sums = league_sums.reset_index(drop=True).drop(0)

league_sums[['xwobacon','max_ev','ev','la']] = 0
for col in league_sums.columns:
    if col in ['pitch_count','max_ev','bip']:
        continue
    elif col in ['xwobacon']:
        league_sums[col][1] = round(mean(fbb.estimated_woba_using_speedangle),3)
    elif col in ['ev']:
        league_sums[col][1] = round(mean(fbb.launch_speed),3)
    elif col in ['la']:
        league_sums[col][1] = round(mean(fbb.launch_angle),3)
    elif col in ['bip']:
        league_sums[col][1] = round(sum(batters.in_play)/len(batters),3)
    else:
        if col == 'barrels':
            o_col = 'barrel'
        elif col == 'poorly_hit':
            o_col = 'weak'
        else:
            o_col = col
        league_sums[col][1] = round(sum(batters[o_col])/len(fbb),3)

league_sums['max_ev'] = round(mean(grouped.query('bip >= 30').max_ev),2)
league_sums.bip = 1
league_sums['pitch_count'] = round(1/0.176,1)
    
    
# now that we have per-pitch averages for the league, we will regress for players with under 30 bip/30 pitches seen
for i in range(len(grouped)):
    if grouped.bip[i] < 30:
            diff = 30 - grouped.bip[i]
            added_sums = round(league_sums.iloc[:,:10] * diff,2)
            grouped.iloc[i:i+1,4:14] += (added_sums.iloc[:,:10]).values
            grouped.iloc[i:i+1,17:21] = (grouped.iloc[i:i+1,17:21]*(grouped.bip[i]/30))+((league_sums.iloc[0:1,12:].values)*(diff/30))
            grouped.pitch_count[i] += round(league_sums.pitch_count[1]*diff)
            grouped.bip[i] = 30
    else:
        continue
import numpy as np

grouped = grouped.drop(grouped.query('fly_ball == 0').index).reset_index(drop=True)
grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])
rates = grouped[['player_name','playerid','game_year','pitch_type','xwobacon','la','ev','max_ev','gb/fb',
                 'pitch_count','bip','swing']]
rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
#split_metrics[[averages.columns[[range(2,len(averages.columns))]]]] = 0
for i in range(0,len(grouped)):
    rates['whiff'][i] = round(grouped['whiff'][i]/grouped['swing'][i],3)*100
    rates['hh'][i] = round(grouped['hh'][i]/grouped['bip'][i],3)*100
    rates['ld'][i] = round((grouped['line_drive'][i])/grouped['bip'][i],3)*100
    rates['hr'][i] = round((grouped['home_run'][i])/grouped['bip'][i],3)*100
    rates['barrel'][i] = round((grouped['barrels'][i])/grouped['bip'][i],3)*100
    rates['weak'][i] = round((grouped['poorly_hit'][i])/grouped['bip'][i],3)*100
    rates['swing%'][i] = round((grouped['swing'][i])/grouped['pitch_count'][i],3)*100
    rates['chase%'][i] = round((grouped['chase'][i])/grouped['swing'][i],3)*100

pitch_list = list(rates['pitch_type'].unique())
rates = rates.reindex(columns=['player_name','playerid','game_year',
                                         'pitch_type','hh','barrel','weak',
                                         'ld','whiff','chase%',
                                         'swing%','hr',
                                         'xwobacon','max_ev','ev','la','bip','pitch_count','gb/fb'])

# get averages for the league environment

just_stats = rates.iloc[:,4:rates.shape[1]].drop(columns=['pitch_count','bip'])
pitch_avgs = pitch_avgs.rename(columns = {'barrels':'barrel','chase':'chase%','home_run':'hr','line_drive':'ld',
                                          'poorly_hit':'weak','swing':'swing%'})
for i in range(0,len(just_stats)):
    pitch = rates.iloc[:,3][i]
    bucket_subset = pitch_avgs.query('pitch_type == @pitch')
    if i == 0:
        new_stats = round(just_stats.loc[i]/(bucket_subset.iloc[:,1:bucket_subset.shape[1]].drop(columns=['pitch_count','bip'])),2)
    else:
        data = round(just_stats.loc[i]/(bucket_subset.iloc[:,1:bucket_subset.shape[1]].drop(columns=['pitch_count','bip'])),2)
        new_stats = new_stats.append(data)
new_stats = new_stats.reset_index(drop=True)
new_stats[['player_name','playerid','game_year','pitch_type','pitch_count','bip']] = rates[['player_name','playerid','game_year','pitch_type','pitch_count','bip']]
new_stats["player_name"] = [" ".join(n.split(", ")[::-1]) for n in new_stats["player_name"]]
new_stats['player_name'] = new_stats['player_name'].apply(unidecode)
new_stats.iloc[:,:13] = new_stats.iloc[:,:13].astype(float).round(2)
#%
names = new_stats[['player_name','playerid']]
names = names.drop_duplicates(subset=['player_name','playerid'], keep='first').reset_index(drop=True)
# full_player_stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
old_hits = pd.read_csv('old_hits.csv')
X = old_hits.iloc[:,3:16].drop(columns='hr')
y = old_hits.hr
result = LassoCV(cv=5, random_state=79, max_iter=10000)
result = result.fit(X, y)


weighted = pd.DataFrame()
for i in range(0,len(names)):
    name = names['playerid'][i]
    player = new_stats.query('playerid == @name').reset_index().drop(columns='index')
    for q in range(0,len(player)):
        for p in range(0,13):
            year = player.game_year[q]
            if p == 9:
                continue
            else:
                player.iloc[:,p][q] = round(player.iloc[:,p][q]*(player['pitch_count'][q]/sum(player[player['game_year'] == year]['pitch_count'])),5)
    player = player.groupby(['player_name','game_year','playerid']).agg({
        **{col: 'sum' for col in list(player.columns[:9])},
        **{col: 'mean' for col in [player.columns[9]]},
        **{col: 'sum' for col in list(player.columns[10:13])},
        **{col: 'sum' for col in list(player.columns[17:])}}).reset_index()
    if len(player['game_year'].unique()) == 2:
        player = player.groupby(['player_name','playerid']).agg({
            **{col: 'mean' for col in list(player.columns[3:16])},
            **{col: 'sum' for col in player.columns[16:]}}).reset_index()
    elif sum(player.bip) < 200 and sum(player.game_year) == 2025:
        p_weight = round(sum(player.bip)/200,2)
        player.iloc[:,2:14] = (player.iloc[:,2:14])*p_weight + (1*(1-p_weight))
    else:
        player.iloc[:,2:14] = (player.iloc[:,2:14]+1)/2
    player = round(player,2)
    weighted = weighted.append(player)
    weighted = weighted.reset_index(drop=True)
    player = round(player,2)
    weighted = weighted.append(player)
weighted = weighted.drop_duplicates().reset_index(drop=True)
weighted = weighted.drop(columns='hr')
weighted['pred_hr'] = result.predict(weighted.iloc[:,2:14]).round(2)




    
per_pitch_short = pd.DataFrame()
options = new_stats[['player_name','playerid','pitch_type','game_year','bip']]
options = options.drop_duplicates(subset=['player_name','playerid','pitch_type'], keep='first').reset_index(drop=True)
new_stats = new_stats.drop(columns='hr')
for i in range(0,len(options)):
    bbe = options['bip'][i]
    name = options['playerid'][i]
    pitch_type = options['pitch_type'][i]
    pitch = new_stats.query('playerid == @name and pitch_type == @pitch_type').reset_index(drop=True)
    player = weighted.query('playerid == @name').reset_index(drop=True)
    if len(pitch) > 1:
    # Get BBE values for both years before grouping
        bbe_2024 = pitch.query('game_year == 2024')['bip'].iloc[0] if len(pitch.query('game_year == 2024')) > 0 else 0
        bbe_2025 = pitch.query('game_year == 2025')['bip'].iloc[0] if len(pitch.query('game_year == 2025')) > 0 else 0
        
        # Calculate weights based on BBE scenarios
        weight_2025 = 0.67 + 0.23 * (bbe_2025 / (bbe_2024 + bbe_2025))       
        weight_2024 = 1 - weight_2025
        pitch_2024 = pitch.query('game_year == 2024').reset_index(drop=True)
        pitch_2025 = pitch.query('game_year == 2025').reset_index(drop=True)
        
        for col_idx in range(12):
            col_name = pitch.columns[col_idx]
            if col_name not in ['player_name', 'playerid', 'pitch_type', 'game_year']:
                val_2024 = pitch_2024[col_name].iloc[0] if len(pitch_2024) > 0 else 0
                val_2025 = pitch_2025[col_name].iloc[0] if len(pitch_2025) > 0 else 0
                weighted_val = (val_2024 * weight_2024 + val_2025 * weight_2025).round(2)
                pitch.loc[0, col_name] = weighted_val
        
        pitch = pitch.iloc[:1].copy()
        for col in pitch.columns[16:]:
            pitch[col] = pitch_2024[col].sum() + pitch_2025[col].sum() if len(pitch_2024) > 0 and len(pitch_2025) > 0 else pitch[col]
    else:
        continue
    if pitch.bip[0] < 100:
        pitch.iloc[:,:12] = (pitch.iloc[:,:12].values * (pitch.bip[0]/100)) + player.iloc[:,2:14].values * ((100-pitch.bip[0])/100)
        pitch = pitch.round(2)
        pitch.bip[0] = 100
        pitch.pitch_count[0] += 5.8*(100-pitch.bip[0])
    per_pitch_short = per_pitch_short.append(pitch)
per_pitch_short = per_pitch_short.reset_index(drop=True)
per_pitch_short['pred_hr'] = result.predict(per_pitch_short.iloc[:,:12]).round(2)
per_pitch_short = per_pitch_short.drop(columns='game_year')

#%% lasso regression with incomplete swing data

# first method - only use data from 2023 onward

bat_stats = weighted.query('game_year >= 2023')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, MultiTaskLassoCV,ElasticNetCV,RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import numpy as np
X = bat_stats.iloc[:,3:21].drop(columns='hr')
y = bat_stats.hr
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

result = LassoCV(alphas=np.logspace(-3, 1, 20),cv=5, random_state=14, max_iter=10000)
result = result.fit(X_train, y_train)
coef_df = pd.DataFrame({'Variable': X_train.columns,'Coefficient': result.coef_})
#
X_test['pred_hr'] = result.predict(X_test).round(2)
X_train['pred_hr'] = result.predict(X_train).round(2)
mse(X_test.pred_hr,y_test)
mse(X_train.pred_hr,y_train)
y_pred = result.predict(X)

pearsonr(y_pred,y) #0.88
# high disparity between train and test MSE, but high pearsonr makes up for it

# testing without swing metrics
no_swing = X.drop(columns=['bat_speed', 'swing_length', 'attack_angle', 'attack_direction', 'swing_path_tilt'])
y = bat_stats.hr
X_train, X_test, y_train, y_test = train_test_split(no_swing, y, test_size=0.20, random_state=12)

result = LassoCV(alphas=np.logspace(-3, 1, 20),cv=5, random_state=14, max_iter=10000)
result = result.fit(X_train, y_train)
coef_df = pd.DataFrame({'Variable': X_train.columns,'Coefficient': result.coef_})
#
X_test['pred_hr'] = result.predict(X_test).round(2)
X_train['pred_hr'] = result.predict(X_train).round(2)
mse(X_test.pred_hr,y_test)
mse(X_train.pred_hr,y_train)
y_pred = result.predict(no_swing)
pearsonr(y_pred,y) #0.877

"""
virtually no change in pearsonr or mse between including swing metrics or no. We will
be removing them to include data from 2015-2022"""

#%% Using all years with no swing metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import numpy as np
nty = weighted.query('game_year != 2025')
X = nty.iloc[:,3:21].drop(columns=['hr','bat_speed','swing_length','attack_angle','attack_direction','swing_path_tilt'])
y = nty.hr
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

result = LassoCV(alphas=np.logspace(-3, 1, 20),cv=5, random_state=14, max_iter=10000)
result = result.fit(X_train, y_train)
coef_df = pd.DataFrame({'Variable': X_train.columns,'Coefficient': result.coef_})
#
X_test['pred_hr'] = result.predict(X_test).round(2)
X_train['pred_hr'] = result.predict(X_train).round(2)
mse(X_test.pred_hr,y_test)
mse(X_train.pred_hr,y_train)
y_pred = result.predict(X)

pearsonr(y_pred,y) #0.90

"""higher pearsonr with lower disparity between train and test mse, we will
not be including bat tracking data, at least for the 2025 iteration of this model"""
nty = pd.read_csv('old_hits.csv')
nty = nty.drop(columns=['bat_speed','swing_length','attack_angle','attack_direction','swing_path_tilt'])
nty.to_csv('old_hits.csv',index=False)
#%% second method - predict bat stats pre-2023 - didn't work
from numpy import nan
swing_cols = ['bat_speed', 'swing_length', 'attack_angle', 'attack_direction', 'swing_path_tilt']

for i in range(len(weighted)):
    if sum(weighted[swing_cols].loc[i]) == 0:
        for col in swing_cols:
            weighted[col].loc[i] = nan

swing_cols = ['bat_speed', 'swing_length', 'attack_angle', 'attack_direction', 'swing_path_tilt']
predictor_cols = [ 'whiff', 'ev', 'la', 'barrel', 'hh','max_ev']  # adjust as needed

# Split data: use 2023+ to predict pre-2023
train_data = weighted[weighted['game_year'] >= 2023].dropna(subset=swing_cols)
predict_data = weighted[weighted['game_year'] < 2023]

X_train = train_data[predictor_cols]
y_train = train_data[swing_cols]  # Multiple targets!

X_predict = predict_data[predictor_cols]

# Method 1: MultiTaskLasso (preferred for related targets)
model = MultiTaskLassoCV(cv=5, random_state=42, max_iter=10000)
model = model.fit(X_train, y_train)
predict_data[swing_cols] = model.predict(predict_data[predictor_cols]).round(2)
weighted = predict_data.append(train_data).reset_index(drop=True)

r2(predict_data.bat_speed,predict_data.hr)
r2(train_data.bat_speed,train_data.hr)
#%%    
    if len(player['year'].unique()) >= 2:
        player = player.groupby(['player_name','playerid']).agg({
            **{col: 'mean' for col in list(player.columns[3:14])},
            **{col: 'sum' for col in player.columns[14:]}}).reset_index()
    else:
        player.iloc[:,3:14] = (player.iloc[:,3:14]+1)/2
    player = round(player,2)
    weighted = weighted.append(player)
weighted = weighted.reset_index(drop=True)
#%% pitchers
import pandas as pd
from numpy import select,nan,inf
from unidecode import unidecode
import numpy as np
from statistics import mean
from math import floor, ceil
pitchers = pd.read_csv('pitchers.csv')
pitchers = pitchers.rename(columns={'game_year':'year'})
# Removing bunts from the analysis, would screw up ev and la metrics

pitchers = pitchers[~(pitchers['description'].str.contains('bunt', case=False))]
pitchers = pitchers[~pitchers['events'].isin(['sac_bunt','sac_bunt_double_play'])]

#remove intent_ball, velo would be messed up

pitchers = pitchers[~(pitchers['description'].str.contains('intent_ball', case=False))]

# distinction between blocked or not and foul_tip/swinging_strike doesn't matter

pitchers['description'] = pitchers['description'].str.replace('swinging_strike_blocked', 'swinging_strike')
pitchers['description'] = pitchers['description'].str.replace('blocked_ball', 'ball')
pitchers['description'] = pitchers['description'].str.replace('foul_tip', 'swinging_strike') 

# discovered that bb_types were not correct so changed them manually

pitchers = pitchers.reset_index().drop(columns='index')
conditions = [
pitchers['launch_angle'].isna(),
pitchers['launch_angle'] < 10,
(pitchers['launch_angle'] >= 10) & (pitchers['launch_angle'] <= 25),
(pitchers['launch_angle'] > 25)]
choices = ['nan','ground_ball','line_drive','fly_ball']
pitchers['bb_type'] = select(conditions, choices) 
pitchers['bb_type'].replace('nan',pitchers['launch_angle'][1], inplace = True)
pitchers = pitchers.reset_index().drop(columns='index')
pitchers['in_play'] = (pitchers['description'] == 'hit_into_play').astype(int)
pitchers['barrel'] = (pitchers['launch_speed_angle'] == 6).astype(int)
pitchers['weak'] = (pitchers['launch_speed_angle'].isin([1,2])).astype(int)
pitchers['fly_ball'] = pitchers.apply(lambda row: 1 if row['bb_type'] == 'fly_ball' and row['description'] != 'foul' else 0, axis = 1)
pitchers['ground_ball'] = pitchers.apply(lambda row: 1 if row['bb_type'] == 'ground_ball' and row['description'] != 'foul' else 0, axis = 1)
pitchers['line_drive'] = pitchers.apply(lambda row: 1 if row['bb_type'] == 'line_drive' and row['description'] != 'foul' else 0, axis = 1)
pitchers['whiff'] = (pitchers['description'] == 'swinging_strike').astype(int)
pitchers['swing'] = (pitchers['description'].isin(['swinging_strike','hit_into_play','foul'])).astype(int)
pitchers['home_run'] = (pitchers['events'] == 'home_run').astype(int)
pitchers['hh'] = (pitchers['launch_speed'] >= 95).astype(int)
pitchers['in_zone'] = (pitchers['zone'] < 10).astype(int)
pitchers['chase'] = pitchers.apply(lambda row: 1 if row['swing'] == 1 and row['in_zone'] == 0 else 0, axis = 1)
pitchers['plate_x'] = abs(pitchers.plate_x)


fbb = pitchers.query('in_play==1')
fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])

grouped = pitchers.groupby(['player_name','playerid', 'year', 'pitch_type']).agg(
    velo=('release_speed', 'mean'),
    spin_rate=('release_spin_rate', 'mean'),
    hh=('hh', 'sum'), # Count of batted balls with exit velocity over 95 mph
    bip=('in_play','sum'),
    barrels =('barrel','sum'),
    poorly_hit=('weak','sum'),
    fly_ball =('fly_ball','sum'),
    ground_ball =('ground_ball','sum'),
    line_drive =('line_drive','sum'),
    whiff =('whiff','sum'),
    chase=('chase','sum'),
    swing =('swing','sum'),
    x_move =('plate_x','mean'),
    z_move =('plate_z','mean'),
    extension =('release_extension','mean'),
    home_run =('home_run','sum'),
    pitch_count=('pitch_type','size')).reset_index()
xgrouped = fbb.groupby(['player_name','playerid', 'year', 'pitch_type']).agg(
tot_wob =('estimated_woba_using_speedangle','sum'),
count=('pitch_type','size')).reset_index()
xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
xgrouped = xgrouped[['player_name','playerid','year','pitch_type','xwobacon']]
grouped = grouped.merge(xgrouped,on=['player_name','playerid','year','pitch_type'])
#grouped = grouped.dropna(subset=['velo','spin_rate'])
grouped = grouped.round({'spin_rate': 0, 'velo': 1, 'rls_avg': 2, 'rls_std': 2})

grouped[['ev', 'la']] = pitchers.groupby(['player_name', 'playerid', 'year', 'pitch_type'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name', 'playerid', 'year', 'pitch_type']).index).values


grouped = grouped.dropna()
pitch_list = list(grouped.pitch_type.unique())
pitch_avgs = grouped.groupby('pitch_type').agg({**{col: 'sum' for col in list(grouped.columns[6:21])}})
pitch_avgs[['velo','spin_rate','x_move','z_move','extension','xwobacon','ev','la']] = 0
for pitch in pitch_list:
    subset = grouped.query('pitch_type == @pitch').reset_index(drop=True)
    for col in ['velo','spin_rate','x_move','z_move','extension','xwobacon','ev','la']:
        total = 0
        for i in range(len(subset)):
            if col == 'max_ev':
                total = mean(subset.max_ev)
            else:
                value = subset[col][i]*(subset.pitch_count[i]/sum(subset.pitch_count)).round(5)
                total += value
        total = round(total,3)
        pitch_avgs[col][pitch] = total
pitch_avgs['gb/fb'] = (pitch_avgs['ground_ball']/pitch_avgs['fly_ball'])
pitch_avgs = pitch_avgs.drop(columns=['fly_ball','ground_ball'])

for col in pitch_avgs.columns[:8]:
    if col == 'bip':
        continue
    elif col in ['chase','whiff']:
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['swing'])*100,2)
    elif col == 'swing':
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['pitch_count'])*100,2)
    elif col=='home_run':
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['bip'])*100,4)
    else:
        pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['bip'])*100,2)
pitch_avgs = pitch_avgs.reset_index()


# getting league averages to regress small sample sizes

league_sums= pd.DataFrame((grouped.groupby('year').agg({
    **{col: 'sum' for col in list(grouped.columns[6:21])}})).sum().reset_index()).T
league_sums.columns = league_sums.loc['index']
league_sums = league_sums.reset_index(drop=True).drop(0)

league_sums[['xwobacon','x_move','z_move','extension','ev','la']] = 0
for col in league_sums.columns:
    if col in ['pitch_count','max_ev','bip']:
        continue
    elif col in ['xwobacon']:
        league_sums[col][1] = round(mean(fbb.estimated_woba_using_speedangle),3)
    elif col in ['x_move','z_move','extension']:
        league_sums[col][1] = 0
    elif col in ['ev']:
        league_sums[col][1] = round(mean(fbb.launch_speed),3)
    elif col in ['la']:
        league_sums[col][1] = round(mean(fbb.launch_angle),3)
    elif col in ['bip']:
        league_sums[col][1] = round(sum(pitchers.in_play)/len(batters),3)
    else:
        if col == 'barrels':
            o_col = 'barrel'
        elif col == 'poorly_hit':
            o_col = 'weak'
        else:
            o_col = col
        league_sums[col][1] = round(sum(pitchers[o_col])/len(fbb),3)


league_sums[['velo','spin_rate','bip','pitch_count']] = 0
grouped = grouped.reset_index(drop=True)

player_sums= pd.DataFrame((grouped.groupby(['player_name','playerid']).agg({
    **{col: 'sum' for col in list(grouped.columns[6:21])}})).reset_index())

player_sums[['xwobacon','x_move','z_move','extension','ev','la']] = 0
for i in range(len(player_sums)):
    name = player_sums.playerid[i]
    for col in player_sums.columns[2:]:
        if col in ['pitch_count','max_ev','bip']:
            continue
        elif col in ['xwobacon']:
            player_sums[col][i] = round(mean(fbb.query('playerid==@name').estimated_woba_using_speedangle),3)
        elif col in ['x_move','z_move','extension']:
            player_sums[col][i] = 0
        elif col in ['ev']:
            player_sums[col][i] = round(mean(fbb.query('playerid==@name').launch_speed),3)
        elif col in ['la']:
            player_sums[col][i] = round(mean(fbb.query('playerid==@name').launch_angle),3)
        else:
            if col == 'barrels':
                o_col = 'barrel'
            elif col == 'poorly_hit':
                o_col = 'weak'
            else:
                o_col = col
            player_sums[col][i] = round(sum(pitchers.query('playerid==@name')[o_col])/len(fbb.query('playerid==@name')),3)

player_sums[['velo','spin_rate','bip','pitch_count']] = 0
grouped = grouped.reset_index(drop=True)
# now that we have per-pitch averages for the league and for each player, we will regress for players with under 30 bip
for i in range(len(grouped)):
    name = grouped.playerid[i]
    if sum(player_sums.query('playerid == @name').bip) < 30:
        if grouped.bip[i] < 30:
            diff = 30 - grouped.bip[i]
            added_sums = round(league_sums.iloc[:,:14] * diff,2)
            grouped.iloc[i:i+1,6:20] += (added_sums.iloc[:,:14]).values
            grouped.iloc[i:i+1,21:25] = (grouped.iloc[i:i+1,21:25]*((30-diff)/30))+((league_sums.iloc[0:1,15:18].values)*(diff/30))
        else:
            continue
    else:
        if grouped.bip[i] < 30:
            diff = 30 - grouped.bip[i]
            p_weight = ceil(diff/2)
            l_weight = floor(diff/2)
            added_sums = round((league_sums.iloc[:,:14] * l_weight)+(league_sums.iloc[:,:14] * p_weight),2)
            grouped.iloc[i:i+1,6:20] += (added_sums.iloc[:,:14]).values
            grouped.iloc[i:i+1,21:25] = (grouped.iloc[i:i+1,21:25]*((30-diff)/30))+((league_sums.iloc[0:1,15:18].values)*(l_weight/30))+((player_sums.iloc[0:1,17:20].values)*(p_weight/30))
        else:
            continue
import numpy as np

# needed average release point data to create metric for release consistency
# have to get rid of this due to pitch minnimums

"""
pitcher_means = pitchers.groupby(['player_name','playerid','pitch_type','year'])[['release_pos_x', 'release_pos_z']].mean()
pitcher_means.columns = ['mean_release_x', 'mean_release_z']
pitchers = pitchers.merge(pitcher_means,on=['player_name','playerid','pitch_type','year'])
pitchers['release_deviation'] = np.sqrt(
   (pitchers['release_pos_x'] - pitchers['mean_release_x'])**2 + 
   (pitchers['release_pos_z'] - pitchers['mean_release_z'])**2
)

grouped = grouped.merge(pitcher_means, on=['player_name','playerid','pitch_type','year'])


# Aggregate to pitcher level
deviations = pitchers.groupby(['player_name','playerid','pitch_type','year'])['release_deviation'].mean().reset_index()
grouped = grouped.merge(deviations,on=['player_name','playerid','pitch_type','year'])

grouped['release_con%'] = round(grouped['release_deviation'].rank(pct=True, ascending=False) * 100,1)
grouped = grouped.drop(columns=['release_deviation','mean_release_x','mean_release_z','release_con%'])
"""
grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])

rates = grouped[['player_name','playerid','year','pitch_type','xwobacon',
             'la','ev','gb/fb','swing','pitch_count','bip','x_move','z_move','extension','velo','spin_rate']]
rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
#split_metrics[[averages.columns[[range(2,len(averages.columns))]]]] = 0
for i in range(0,len(grouped)):
    if grouped['bip'][i] < 30:
        bip = 30
        pitch = grouped['pitch_count'][i] + round((bip-grouped['bip'][i])*5.8)
        rates['whiff'][i] = round(grouped['whiff'][i]/grouped['swing'][i],3)*100
        rates['hh'][i] = round(grouped['hh'][i]/bip,3)*100
        rates['ld'][i] = round((grouped['line_drive'][i])/bip,3)*100
        rates['hr'][i] = round((grouped['home_run'][i])/bip,3)*100
        rates['barrel'][i] = round((grouped['barrels'][i])/bip,3)*100
        rates['weak'][i] = round((grouped['poorly_hit'][i])/bip,3)*100
        rates['swing%'][i] = round((grouped['swing'][i])/pitch,3)*100
        rates['chase%'][i] = round((grouped['chase'][i])/grouped['swing'][i],3)*100
    else:    
        rates['whiff'][i] = round(grouped['whiff'][i]/grouped['swing'][i],3)*100
        rates['hh'][i] = round(grouped['hh'][i]/grouped['bip'][i],3)*100
        rates['ld'][i] = round((grouped['line_drive'][i])/grouped['bip'][i],3)*100
        rates['hr'][i] = round((grouped['home_run'][i])/grouped['bip'][i],3)*100
        rates['barrel'][i] = round((grouped['barrels'][i])/grouped['bip'][i],3)*100
        rates['weak'][i] = round((grouped['poorly_hit'][i])/grouped['bip'][i],3)*100
        rates['swing%'][i] = round((grouped['swing'][i])/grouped['pitch_count'][i],3)*100
        rates['chase%'][i] = round((grouped['chase'][i])/grouped['swing'][i],3)*100




rates = rates.replace(inf, nan)
rates = rates.dropna().reset_index(drop=True)
pitch_list = list(rates['pitch_type'].unique())
test= rates.reindex(columns=['player_name','playerid','year',
                                     'pitch_type','velo','spin_rate','hh','barrel','weak',
                                     'ld','whiff','chase%','swing%', 'x_move',
                                     'z_move','extension',
                                     'hr','xwobacon','ev','la','gb/fb',
                                     'bip','swing','pitch_count'])

pitch_avgs = pitch_avgs.reindex(columns=['pitch_type','velo','spin_rate','hh','barrels','poorly_hit',
                                     'line_drive','whiff','chase','swing', 'x_move',
                                     'z_move','extension',
                                     'home_run','xwobacon','ev','la','gb/fb',
                                     'bip','pitch_count'])

just_stats = rates.iloc[:,4:rates.shape[1]].drop(columns=['pitch_count','bip','swing'])
pitch_avgs = pitch_avgs.rename(columns = {'barrels':'barrel','chase':'chase%','home_run':'hr','line_drive':'ld',
                                      'poorly_hit':'weak','swing':'swing%'})

for i in range(0,len(just_stats)):
    pitch = rates.iloc[:,3][i]
    bucket_subset = pitch_avgs.query('pitch_type == @pitch')
    if i == 0:
        new_stats = round(just_stats.loc[i]/(bucket_subset.iloc[:,1:bucket_subset.shape[1]].drop(columns=['pitch_count','bip'])),2)
    else:
        data = round(just_stats.loc[i]/(bucket_subset.iloc[:,1:bucket_subset.shape[1]].drop(columns=['pitch_count','bip'])),2)
        new_stats = new_stats.append(data)
new_stats = new_stats.reset_index(drop=True)
new_stats[['player_name','playerid','year','pitch_type','pitch_count','bip']] = rates[['player_name','playerid','year','pitch_type','pitch_count','bip']]
new_stats["player_name"] = [" ".join(n.split(", ")[::-1]) for n in new_stats["player_name"]]
new_stats["player_name"] =new_stats["player_name"].apply(unidecode)


# EDIT ILOCS

names = new_stats[['player_name','playerid']]
names = names.drop_duplicates(subset=['player_name','playerid'], keep='first').reset_index(drop=True)
weighted = pd.DataFrame()
for i in range(len(names)):
    name = names['playerid'][i]
    player = new_stats.query('playerid == @name').reset_index().drop(columns='index')
    for q in range(0,len(player)):
        year = player['year'][q]
        for p in range(0,17):
            player.iloc[:,p][q] = round(player.iloc[:,p][q]*(player['pitch_count'][q]/sum(player[player['year'] == year]['pitch_count'])),5)
    player = player.groupby(['player_name','year','playerid']).agg({
        **{col: 'sum' for col in list(player.columns[:17])},
        **{col: 'sum' for col in list(player.columns[21:])}}).reset_index()
    player = round(player,2)
    weighted = weighted.append(player)
weighted = weighted.reset_index(drop=True) #had to drop severe outlier. gb/fb was 31.6 while next highest was 6.3

# Using all years with no swing metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import numpy as np
old_pitch = pd.read_csv('old_pitch.csv')
X = old_pitch.iloc[:,3:20].drop(columns='hr')
y = old_pitch.hr
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)
result = LassoCV(alphas=np.logspace(-3, 1, 20),cv=5, random_state=14, max_iter=10000)
result = result.fit(X_train, y_train)
#
per_pitch_short = pd.DataFrame()
options = new_stats[['player_name','playerid','pitch_type','year','bip']]
options = options.drop_duplicates(subset=['player_name','playerid','pitch_type'], keep='first').reset_index(drop=True)
new_stats = new_stats.drop(columns='hr')
for i in range(0,len(options)):
    bbe = options['bip'][i]
    name = options['playerid'][i]
    pitch_type = options['pitch_type'][i]
    pitch = new_stats.query('playerid == @name and pitch_type == @pitch_type').reset_index(drop=True)
    player = weighted.query('playerid == @name').reset_index(drop=True).drop(columns='hr')
    if len(pitch) > 1:
    # Get BBE values for both years before grouping
        bbe_2024 = max(30,pitch.query('year == 2024')['bip'].iloc[0] if len(pitch.query('year == 2024')) > 0 else 0)
        bbe_2025 = max(30,pitch.query('year == 2025')['bip'].iloc[0] if len(pitch.query('year == 2025')) > 0 else 0)
        
        # Calculate weights based on BBE scenarios
        weight_2025 = 0.67 + 0.23 * (bbe_2025 / (bbe_2024 + bbe_2025))       
        weight_2024 = 1 - weight_2025
        pitch_2024 = pitch.query('year == 2024').reset_index(drop=True)
        pitch_2025 = pitch.query('year == 2025').reset_index(drop=True)
        
        for col_idx in range(15):
            col_name = pitch.columns[col_idx]
            if col_name not in ['player_name', 'playerid', 'pitch_type', 'game_year']:
                val_2024 = pitch_2024[col_name].iloc[0] if len(pitch_2024) > 0 else 0
                val_2025 = pitch_2025[col_name].iloc[0] if len(pitch_2025) > 0 else 0
                weighted_val = (val_2024 * weight_2024 + val_2025 * weight_2025)
                pitch.loc[0, col_name] = weighted_val
        
        pitch = pitch.iloc[:1].copy()
        if sum(pitch_2025.pitch_count) >= 200:
            for col in pitch.columns[20:]:
               pitch[col] = pitch_2025[col]
        else:
            for col in pitch.columns[20:]:
                pitch[col] = pitch_2024[col].sum() + pitch_2025[col].sum() if len(pitch_2024) > 0 and len(pitch_2025) > 0 else pitch[col]
    else:
        continue
    per_pitch_short = per_pitch_short.append(pitch)
per_pitch_short = per_pitch_short.reset_index(drop=True)
per_pitch_short['pred_hr'] = result.predict(per_pitch_short[old_pitch.iloc[:,3:20].drop(columns='hr').columns]).round(2)
per_pitch_short = per_pitch_short.drop(columns='year')
