# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 14:46:16 2025

@author: Brendan
"""

import pandas as pd
batters = pd.read_csv('batters.csv')
fbb = batters.query('in_play ==1').dropna(subset=['estimated_woba_using_speedangle'])

grouped = batters.groupby(['player_name','playerid', 'game_year']).agg(
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

xgrouped = fbb.groupby(['player_name','playerid', 'game_year']).agg(
tot_wob =('estimated_woba_using_speedangle','sum'),
max_ev=('launch_speed','max'),
count=('player_name','size')).reset_index()
xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
xgrouped = xgrouped[['player_name','playerid','game_year','xwobacon','max_ev']]
grouped = grouped.merge(xgrouped,on=['player_name','playerid', 'game_year'])
grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name','playerid', 'game_year']).index).values

newg = grouped.query('bip >= 301').reset_index(drop=True)
newg['gb/fb'] = (newg['ground_ball']/newg['fly_ball'])
full_rates = newg[['player_name','playerid','game_year','xwobacon','la','ev','max_ev','gb/fb',
                 'pitch_count','bip','swing']]
full_rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
full_rates['whiff'] = round(newg['whiff']/newg['swing'],3)*100
full_rates['hh'] = round(newg['hh']/newg['bip'],3)*100
full_rates['ld'] = round((newg['line_drive'])/newg['bip'],3)*100
full_rates['hr'] = round((newg['home_run'])/newg['bip'],3)*100
full_rates['barrel'] = round((newg['barrels'])/newg['bip'],3)*100
full_rates['weak'] = round((newg['poorly_hit'])/newg['bip'],3)*100
full_rates['swing%'] = round((newg['swing'])/newg['pitch_count'],3)*100
full_rates['chase%'] = round((newg['chase'])/newg['swing'],3)*100
del fbb,grouped,xgrouped

from scipy.stats import pearsonr
pop_test = full_rates.drop(columns=['pitch_count','bip','swing'])
samples = pd.Series([50,75,100,150,200,225,250,300])
results = pd.DataFrame()
for k in range(len(samples)):
    i = samples[k]
    raw_pitches = pd.DataFrame()
    for q in range(len(newg)):
        bip = 0
        name = newg.playerid[q]
        year = newg.game_year[q]
        data = batters.query('playerid == @name and game_year ==@year')
        data = data.sort_values(by='game_date',ascending=True).reset_index(drop=True)
        row = (data['in_play'].cumsum() == i).idxmax()+1
        data = data.iloc[:row]
        raw_pitches = raw_pitches.append([data]).reset_index(drop=True)
    fbb = raw_pitches.query('in_play ==1')
    fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])

    grouped = raw_pitches.groupby(['player_name','playerid', 'game_year']).agg(
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
    
    xgrouped = fbb.groupby(['player_name','playerid', 'game_year']).agg(
    tot_wob =('estimated_woba_using_speedangle','sum'),
    max_ev=('launch_speed','max'),
    count=('player_name','size')).reset_index()
    xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
    xgrouped = xgrouped[['player_name','playerid','game_year','xwobacon','max_ev']]
    grouped = grouped.merge(xgrouped,on=['player_name','playerid', 'game_year'])
    grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name','playerid', 'game_year']).index).values
    grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])
    rates = grouped[['player_name','playerid','game_year','xwobacon','la','ev','max_ev','gb/fb',
                     'pitch_count','bip','swing']]
    rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
    
    for p in range(0,len(grouped)):   
        rates['whiff'][p] = round(grouped['whiff'][p]/grouped['swing'][i],3)*100
        rates['hh'][p] = round(grouped['hh'][p]/grouped['bip'][i],3)*100
        rates['ld'][p] = round((grouped['line_drive'][p])/grouped['bip'][i],3)*100
        rates['hr'][p] = round((grouped['home_run'][p])/grouped['bip'][i],3)*100
        rates['barrel'][p] = round((grouped['barrels'][p])/grouped['bip'][i],3)*100
        rates['weak'][p] = round((grouped['poorly_hit'][p])/grouped['bip'][i],3)*100
        rates['swing%'][p] = round((grouped['swing'][p])/grouped['pitch_count'][i],3)*100
        rates['chase%'][p] = round((grouped['chase'][p])/grouped['swing'][i],3)*100 
    samp_test = rates.drop(columns=['pitch_count','bip','swing'])
    for w in range(3,samp_test.shape[1]):
        correl = round(pearsonr(samp_test.iloc[:,w].values,pop_test.iloc[:,w].values)[0],3)
        stat = samp_test.columns[w]
        sample = i
        mush = pd.DataFrame({'stat': [stat], 'sample': [sample], 'correl': [correl]})
        results = results.append(mush)
results = results.reset_index(drop=True)        
results_wide = results.pivot(index='stat', columns='sample', values='correl')
results_change = results_wide.copy()

for i in range(1,len(results_change.columns)):
    results_change.iloc[:, i] = ((results_wide.iloc[:, i] - results_wide.iloc[:, i-1]) / results_wide.iloc[:, i-1]).round(3)
results_wide.columns[0]


"""
Stabilization Points(BBE's) + Confidence Levels:

Asterisk indicates stat denominator is not BBE's

Max EV: 75
    50: 90%
**Whiff Rate: 100
    50: 83%
    75: 94%
Hard-Hit Rate: 100
    50: 81%
    75: 92%
**Chase Rate: 100
    50: 75%
    75: 92%
Barrel Rate: 100
    50: 79%
    75: 91%
Avg EV: 100
    50: 77%
    75: 91%
**Swing Rate: 150
    50: 60%
    75: 83%
    100: 96%
HR Rate: 200
    50: 54%
    75: 65%
    100: 74%
    150: 91%
Weak Contact %: 200
    50: 56%
    75: 69%
    100: 79%
    150: 92%
xwobacon: 200
    50: 60%
    75: 70%
    100: 81%
    150: 92%
Avg LA: 200
    50: 64%
    75: 74%
    100: 83%
    150: 94%
gb/fb rate: 225
    50: 48%
    75: 60%
    100: 69%
    150: 86%
    200: 97%
Line Drive Rate: 250
    50: 35%
    75: 42%
    100: 51%
    150: 70%
    200: 86%
    225: 95%

"""
