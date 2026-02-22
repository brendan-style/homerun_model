# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:37:36 2025

@author: Brendan
"""

"""
in this file, I will be using prior year and current year data to determine
how much the prior year should be weighed. To do this, I am getting all seasons
in the past 5 years where a player had at least 200 bbe's one year, then at 
least 400 BBEs the next. Using this, we can calculate three different sets
of numbers: Prior year, current year first 200 BBEs, and current year post 200
bbes. 200 seems to be the sweet spot where all of our stats are mostly standardized,
so that is why 400 has been selected. 

USE MOST RECENT UP UNTIL SPLIT FOR CURRENT YEAR PRIOR

"""

# first section is standard, just calculating stats for all seasons we have data on
#%% BATTERS
import pandas as pd
from random import randint
from statistics import mean
from numpy import sqrt
import numpy as np
from math import ceil
from scipy.stats import pearsonr
from scipy.optimize import minimize
from numpy import nan, inf
batters = pd.read_csv('batters.csv')
fbb = batters.query('in_play ==1')
fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
grouped = batters.groupby(['player_name','playerid', 'game_year']).agg(
hh=('hh', 'sum'),
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

grouped = grouped.query('fly_ball != 0 and ground_ball != 0')

loop_progress = 0
# getting proper consective seasons
final_results = pd.DataFrame()
pa_count = [10,25,50,75,100,150,200,0]
for loop in range(5,len(pa_count)):
    pa_two = pa_count[loop]
    for u in range(len(pa_count)):
        pa_p = pa_count[u]
        for q in range(len(pa_count)):
            if pa_two == 0 or pa_p == 0:
                continue
            loop_progress += 1
# these 3 nested for loops will allow us to get every combination of BBE total for three seasons
            pa_c = pa_count[q] + 200
            newg = pd.DataFrame()
            years = list(grouped.game_year.unique())
            years.sort(reverse=True)
# gets all data for players who meet the qualifications
            for y in range(len(years)-2):
                c_year = years[y]
                current = grouped.query('bip >= @pa_c and game_year == @c_year').reset_index(drop=True)
                p_year = years[y+1]
                past = grouped.query('bip >= @pa_p and game_year == @p_year').reset_index(drop=True)
                current = current.append(past)
                p2_year = years[y+2]
                past_2 = grouped.query('bip >= @pa_two and game_year == @p2_year').reset_index(drop=True)
                current = current.append(past_2)
                names = current['playerid'].value_counts().reset_index().query('playerid == 3')
                data = grouped[grouped['playerid'].isin(names['index']) & grouped['game_year'].isin(current['game_year'].unique())]
                newg = newg.append(data)
# turns those cumulative stats into rates
            newg = newg.drop_duplicates(subset=['player_name','playerid','game_year'])
            newg = newg.sort_values(by=['playerid','game_year'],ascending=True).reset_index(drop=True)
            newg['gb/fb'] = round(newg['ground_ball']/newg['fly_ball'],2)
            rates2 = newg[['player_name','playerid','game_year','xwobacon','la','ev','max_ev','gb/fb',
                             'pitch_count','bip','swing']]
            rates2[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
            rates2['whiff'] = round(newg['whiff']/newg['swing'],3)*100
            rates2['hh'] = round(newg['hh']/newg['bip'],3)*100
            rates2['ld'] = round((newg['line_drive'])/newg['bip'],3)*100
            rates2['hr'] = round((newg['home_run'])/newg['bip'],3)*100
            rates2['barrel'] = round((newg['barrels'])/newg['bip'],3)*100
            rates2['weak'] = round((newg['poorly_hit'])/newg['bip'],3)*100
            rates2['swing%'] = round((newg['swing'])/newg['pitch_count'],3)*100
            rates2['chase%'] = round((newg['chase'])/newg['swing'],3)*100
            rates2 = rates2.drop(columns=['pitch_count','swing']).reset_index(drop=True)
            rates2 = rates2.sort_values(by=['playerid','game_year'],ascending=True).reset_index(drop=True)
            past_data = pd.DataFrame()
# this for loop does 3 things
            for z in range(len(rates2)-2):
            # checks that the players being evaluated are the same
                name = rates2.playerid[z]
                year_2 = rates2.game_year[z]
                year = rates2.game_year[z+1]
                name_2 = rates2.playerid[z+1]
                name_3 = rates2.playerid[z+2]
                if name == name_2 == name_3:
                    bbe_test = rates2.bip[z+2]
                    if bbe_test < pa_c:
                        continue
                    else:
                        pass
                else:
                    continue
            # if both true, takes their data and grabs random bbe from both "prior" years
                data_2 = batters.query('playerid == @name and game_year == @year_2').reset_index(drop=True)
                data_2['years_ago'] = 2
                data_1 = batters.query('playerid == @name and game_year == @year').reset_index(drop=True)
                data_1['years_ago'] = 1
                if sum(data_2.in_play) < pa_two or sum(data_1.in_play) < pa_p:
                    continue
                else:
                    pass
                for data in [data_2,data_1]:
                    if data.years_ago[0] == 1:
                        samp = pa_p
                    else:
                        samp = pa_two
                    if samp == 0:
                        continue
                    else:
                        pass
                    if sum(data.in_play) == samp:
                        past_data = past_data.append(data)
                        continue
                    data_index = np.arange(len(data))
                    np.random.shuffle(data_index)
            # average pitches per bbe is 5.73
                    total = ceil(5.7*samp)
                    select_rows = data_index[:total]
                    subset = data.iloc[select_rows].copy().reset_index(drop=True)
                    add_on = 2
                    while sum(subset.in_play) != samp:
                        if sum(subset.in_play) > samp:
                            subset = subset.iloc[:-1]
                        elif sum(subset.in_play) < samp:
                            subset = subset.append(data.iloc[data_index[total+(add_on-1):total+add_on]])
                            add_on += 1
                    past_data = past_data.append(subset)
            past_data = past_data.reset_index(drop=True)
            past_data = past_data.drop_duplicates()
            
# since we grabbed the raw pitches, need to transform into rates again
            
            fbb = past_data.query('in_play ==1')
            fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
            past_grouped = past_data.groupby(['player_name','playerid', 'game_year','years_ago']).agg(
            hh=('hh', 'sum'),
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
            xgrouped = fbb.groupby(['player_name','playerid', 'game_year','years_ago']).agg(
            tot_wob =('estimated_woba_using_speedangle','sum'),
            max_ev=('launch_speed','max'),
            count=('player_name','size')).reset_index()
            xgrouped['xwobacon'] = round(xgrouped['tot_wob'].astype(float)/xgrouped['count'],3)
            xgrouped = xgrouped[['player_name','playerid','game_year','years_ago','xwobacon','max_ev']]
            past_grouped = past_grouped.merge(xgrouped,on=['player_name','playerid', 'game_year','years_ago'])
            past_grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(past_grouped.set_index(['player_name','playerid', 'game_year']).index).values
            past_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']] = past_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']].astype(int)
            past_grouped['gb/fb'] = round(past_grouped['ground_ball']/past_grouped['fly_ball'],2)
            past_rates = past_grouped[['player_name','playerid','game_year','years_ago','xwobacon','la','ev','max_ev','gb/fb',
                             'pitch_count','bip','swing']]
            past_rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0  
            past_rates['whiff'] = round(past_grouped['whiff']/past_grouped['swing'],3)*100
            past_rates['hh'] = round(past_grouped['hh']/past_grouped['bip'],3)*100
            past_rates['ld'] = round((past_grouped['line_drive'])/past_grouped['bip'],3)*100
            past_rates['hr'] = round((past_grouped['home_run'])/past_grouped['bip'],3)*100
            past_rates['barrel'] = round((past_grouped['barrels'])/past_grouped['bip'],3)*100
            past_rates['weak'] = round((past_grouped['poorly_hit'])/past_grouped['bip'],3)*100
            past_rates['swing%'] = round((past_grouped['swing'])/past_grouped['pitch_count'],3)*100
            past_rates['chase%'] = round((past_grouped['chase'])/past_grouped['swing'],3)*100
            past_rates = past_rates.drop(columns=['pitch_count','swing'])
            
            
# similar process with current year data, less tricky b/c no random sample needed
            first_x = pd.DataFrame()
            after_200 = pd.DataFrame()
            names = list(rates2.playerid.unique())

            for x in range(len(names)):
                    name = names[x]
                    subset = rates2.query('playerid == @name').reset_index(drop=True)
                    for y in range(2,len(subset)):
                        bip = subset.bip[y]
                        bip_p = subset.bip[y-1]
                        bip_t = subset.bip[y-2]
                        if bip >= pa_c and bip_p >= pa_p and bip_t >= pa_two:
                            if pa_c > 200:
                                year = subset.game_year[y]
                                check = batters.query('playerid == @name and game_year == @year')
                                check = check.sort_values(by='game_date',ascending = True).reset_index(drop=True)
                                row = (check['in_play'].cumsum() == (pa_c-200)).idxmax()+1
                                prior = check.iloc[:row]
                                post = check.iloc[row:]
                                first_x = first_x.append(prior)
                                after_200 = after_200.append(post)
                            else:
                                year = subset.game_year[y]
                                check = batters.query('playerid == @name and game_year == @year')
                                check = check.sort_values(by='game_date',ascending = True).reset_index(drop=True)
                                row = (check['in_play'].cumsum() == (200)).idxmax()+1
                                post = check.iloc[row:]
                                after_200 = after_200.append(post)
                        else:
                            continue
            if not first_x.empty:
                fbb = first_x.query('in_play ==1')
                fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
                first_grouped = first_x.groupby(['player_name','playerid', 'game_year']).agg(
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
                
                first_grouped = first_grouped.query('ground_ball != 0 and fly_ball != 0').reset_index(drop=True)
                
                xgrouped = fbb.groupby(['player_name','playerid', 'game_year']).agg(
                tot_wob =('estimated_woba_using_speedangle','sum'),
                max_ev=('launch_speed','max'),
                count=('player_name','size')).reset_index()
                xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
                xgrouped = xgrouped[['player_name','playerid','game_year','xwobacon','max_ev']]
                first_grouped = first_grouped.merge(xgrouped,on=['player_name','playerid', 'game_year'])
                first_grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(first_grouped.set_index(['player_name','playerid', 'game_year']).index).values
                
                first_grouped['gb/fb'] = round(first_grouped['ground_ball']/first_grouped['fly_ball'],2)
                first_rates = first_grouped[['player_name','playerid','game_year','xwobacon','la','ev','max_ev','gb/fb',
                                 'pitch_count','bip','swing']]
                first_rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
                first_rates['whiff'] = round(first_grouped['whiff']/first_grouped['swing'],3)*100
                first_rates['hh'] = round(first_grouped['hh']/first_grouped['bip'],3)*100
                first_rates['ld'] = round((first_grouped['line_drive'])/first_grouped['bip'],3)*100
                first_rates['hr'] = round((first_grouped['home_run'])/first_grouped['bip'],3)*100
                first_rates['barrel'] = round((first_grouped['barrels'])/first_grouped['bip'],3)*100
                first_rates['weak'] = round((first_grouped['poorly_hit'])/first_grouped['bip'],3)*100
                first_rates['swing%'] = round((first_grouped['swing'])/first_grouped['pitch_count'],3)*100
                first_rates['chase%'] = round((first_grouped['chase'])/first_grouped['swing'],3)*100
                first_rates = first_rates.drop(columns=['pitch_count','swing','bip'])
            else:
                pass
            
            fbb = after_200.query('in_play ==1')
            fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
            after_grouped = after_200.groupby(['player_name','playerid', 'game_year']).agg(
            hh=('hh', 'sum'),
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
            
            after_grouped = after_grouped[after_grouped['playerid'].isin(first_grouped['playerid'])].reset_index(drop=True)
            xgrouped = fbb.groupby(['player_name','playerid', 'game_year']).agg(
            tot_wob =('estimated_woba_using_speedangle','sum'),
            max_ev=('launch_speed','max'),
            count=('player_name','size')).reset_index()
            xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
            xgrouped = xgrouped[['player_name','playerid','game_year','xwobacon','max_ev']]
            after_grouped = after_grouped.merge(xgrouped,on=['player_name','playerid', 'game_year'])
            after_grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(after_grouped.set_index(['player_name','playerid', 'game_year']).index).values

            after_grouped['gb/fb'] = round(after_grouped['ground_ball']/after_grouped['fly_ball'],2)
            after_rates = after_grouped[['player_name','playerid','game_year','xwobacon','la','ev','max_ev','gb/fb',
                             'pitch_count','bip','swing']]
            after_rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
            after_rates['whiff'] = round(after_grouped['whiff']/after_grouped['swing'],3)*100
            after_rates['hh'] = round(after_grouped['hh']/after_grouped['bip'],3)*100
            after_rates['ld'] = round((after_grouped['line_drive'])/after_grouped['bip'],3)*100
            after_rates['hr'] = round((after_grouped['home_run'])/after_grouped['bip'],3)*100
            after_rates['barrel'] = round((after_grouped['barrels'])/after_grouped['bip'],3)*100
            after_rates['weak'] = round((after_grouped['poorly_hit'])/after_grouped['bip'],3)*100
            after_rates['swing%'] = round((after_grouped['swing'])/after_grouped['pitch_count'],3)*100
            after_rates['chase%'] = round((after_grouped['chase'])/after_grouped['swing'],3)*100
            after_rates = after_rates.drop(columns=['pitch_count','swing'])
            after_rates = after_rates.drop(columns='bip')
            past_rates = past_rates.drop(columns='bip')
            
            for i in range(3,after_rates.shape[1]):
                if first_x.empty:
                    column = after_rates.columns[i]
                    merge = after_rates[['player_name','playerid','game_year',column]].sort_values(by=['playerid','game_year'],ascending=False).reset_index(drop=True)
                    column_b = column+'_x'
                    column_a = column+'_y'
                    column_c = column+'_2'
                else:
                    column = past_rates.columns[i+1]
                    first = first_rates[['player_name','playerid','game_year',column]]
                    after = after_rates[['player_name','playerid','game_year',column]]
                    merge = first.merge(after,how='inner',on=['playerid','game_year']).sort_values(by=['playerid','game_year'],ascending=False).reset_index(drop=True)
                    column_b = column+'_x'
                    column_a = column+'_y'
                    column_c = column+'_2'
                for z in range(len(merge)):
                    year1 = (merge.game_year[z])-1
                    year2 = (merge.game_year[z])-2
                    name = merge.playerid[z]
                    data = past_rates.query('playerid == @name and ((game_year == @year1 and years_ago == 1) or (game_year == @year2 and years_ago == 2))')[['player_name','playerid','game_year',column]]
                    if z == 0:
                        past = data
                    else:
                        past = past.append(data)
                past = past.sort_values(by=['playerid','game_year'],ascending=False).reset_index(drop=True)
                evens = past.iloc[::2].reset_index(drop=True)
                odds = past.iloc[1::2].reset_index(drop=True)
                evens[f'{column_c}'] = odds[column]
                evens.game_year = evens.game_year + 1
                merge = merge.merge(evens,how='inner',on=['playerid','game_year']).sort_values(by=['playerid','game_year'],ascending=False).reset_index(drop=True)
                gb_check = merge[merge.isin([inf,nan]).any(axis=1)]
                if gb_check.empty:
                    pass
                else:
                    merge = merge.drop(gb_check.index).reset_index(drop=True)
                if not first_x.empty:
                    def objective(weights):
                        predictions = weights[0] * merge[column_b] + weights[1] * merge[column] + weights[2] * merge[column_c]
                        return -pearsonr(predictions, merge[column_a])[0]
                    
                    result = minimize(objective, x0=[0.33, 0.33, 0.34], 
                                      bounds=[(0, 1), (0, 1), (0, 1)],
                                      constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
                    
                    best_weights = result.x.round(3)
                    if i == 3:
                        results = pd.DataFrame([[column,best_weights[0],best_weights[1],best_weights[2],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])
                    else:
                        results = pd.concat([results,pd.DataFrame([[column,best_weights[0],best_weights[1],best_weights[2],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])],ignore_index=True)
            
                else:
                    def objective(weights):
                        predictions = weights[0] * merge[column_a] + weights[1] * merge[column_c]
                        return -pearsonr(predictions, merge[column_b])[0]
                    
                    result = minimize(objective, x0=[.5,.5], 
                                      bounds=[(0, 1), (0, 1)],
                                      constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
                    
                    best_weights = result.x.round(2)
                    if i == 3:
                        results = pd.DataFrame([[column,0,best_weights[0],best_weights[1],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])
                    else:
                        results = pd.concat([results,pd.DataFrame([[column,0,best_weights[0],best_weights[1],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])],ignore_index=True)
            results[['pre_split_bbe','prior_bbe','two_years_bbe','data_points']] = (pa_c)-200,pa_p,pa_two,len(merge)
            final_results = final_results.append(results)
            final_results.to_csv('bbe_weights.csv',index=False)
"""
In case no gb or fb ever becomes an issue

                        elif sum(subset.in_play) == 100 and sum(subset.fly_ball) == 0:
                            gbd = data.query('fly_ball == 1').reset_index(drop=True)
                            row = randint(0,len(gbd)-1)
                            subset = subset.append(data.iloc[row])
                        elif sum(subset.in_play) == 100 and sum(subset.ground_ball) == 0:
                            gbd = data.query('ground_ball == 1').reset_index(drop=True)
                            row = randint(0,len(gbd)-1)
                            subset = subset.append(data.iloc[row])
"""
#%% PITCHERS
# much tricker b/c a lot of stats are pitch specific
import pandas as pd
from random import randint
from statistics import mean
from numpy import sqrt
import numpy as np
from numpy import inf,nan
from math import ceil,floor
from scipy.stats import pearsonr
from scipy.optimize import minimize
pitchers = pd.read_csv('pitchers.csv')
fbb = pitchers.query('in_play==1')
fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
grouped = pitchers.groupby(['player_name','playerid', 'game_year', 'pitch_type']).agg(
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
    x_move =('pfx_x','mean'),
    z_move =('pfx_z','mean'),
    extension =('release_extension','mean'),
    home_run =('home_run','sum'),
    pitch_count=('pitch_type','size')).reset_index()
xgrouped = fbb.groupby(['player_name','playerid', 'game_year', 'pitch_type']).agg(
tot_wob =('estimated_woba_using_speedangle','sum'),
count=('pitch_type','size')).reset_index()
xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
xgrouped = xgrouped[['player_name','playerid','game_year','pitch_type','xwobacon']]
grouped = grouped.merge(xgrouped,on=['player_name','playerid','game_year','pitch_type'])
#grouped = grouped.dropna(subset=['velo','spin_rate'])
grouped = grouped.round({'spin_rate': 0, 'velo': 1, 'rls_avg': 2, 'rls_std': 2})

grouped[['ev', 'la']] = pitchers.groupby(['player_name', 'playerid', 'game_year', 'pitch_type'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name', 'playerid', 'game_year', 'pitch_type']).index).values
grouped = grouped.query('fly_ball != 0 and ground_ball != 0')
grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])
loop_progress = 0
final_results = pd.DataFrame()
pa_count = [10,25,50,75,100,0]
for loop in range(0,len(pa_count)):
    pa_two = pa_count[loop]
    for u in range(len(pa_count)):
        pa_p = pa_count[u]
        for q in range(len(pa_count)):
            if pa_two == 0 or pa_p == 0:
                continue
            loop_progress += 1
# these 3 nested for loops will allow us to get every combination of BBE total for three seasons
            pa_c = pa_count[q] + 100
            newg = pd.DataFrame()
            years = list(grouped.game_year.unique())
            years.sort(reverse=True)
# gets all data for players who meet the qualifications
            for y in range(len(years)-2):
                c_year = years[y]
                current = grouped.query('bip >= @pa_c and game_year == @c_year').reset_index(drop=True)
                p_year = years[y+1]
                past = grouped.query('bip >= @pa_p and game_year == @p_year').reset_index(drop=True)
                current = current.append(past)
                p2_year = years[y+2]
                past_2 = grouped.query('bip >= @pa_two and game_year == @p2_year').reset_index(drop=True)
                current = current.append(past_2)
                names = current[['playerid','pitch_type']].value_counts().reset_index().rename(columns={0:'id_check'}).query('id_check == 3')
                data = grouped[grouped['playerid'].isin(names['playerid']) & grouped['game_year'].isin(current['game_year'].unique())]
                newg = newg.append(data)
# turns those cumulative stats into rates
            newg = newg.drop_duplicates(subset=['player_name','playerid','game_year','pitch_type'])
            newg = newg.sort_values(by=['playerid','pitch_type','game_year'],ascending=True).reset_index(drop=True)
            newg['gb/fb'] = round(newg['ground_ball']/newg['fly_ball'],2)
            rates2 = newg[['player_name','playerid','game_year','pitch_type','xwobacon',
                 'la','ev','gb/fb','swing','pitch_count','bip','x_move','z_move','extension','velo','spin_rate']]
            rates2[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
            rates2['whiff'] = round(newg['whiff']/newg['swing'],3)*100
            rates2['hh'] = round(newg['hh']/newg['bip'],3)*100
            rates2['ld'] = round((newg['line_drive'])/newg['bip'],3)*100
            rates2['hr'] = round((newg['home_run'])/newg['bip'],3)*100
            rates2['barrel'] = round((newg['barrels'])/newg['bip'],3)*100
            rates2['weak'] = round((newg['poorly_hit'])/newg['bip'],3)*100
            rates2['swing%'] = round((newg['swing'])/newg['pitch_count'],3)*100
            rates2['chase%'] = round((newg['chase'])/newg['swing'],3)*100
            rates2 = rates2.drop(columns=['pitch_count','swing']).reset_index(drop=True)
            rates2 = rates2.sort_values(by=['playerid','pitch_type','game_year'],ascending=True).reset_index(drop=True)
            past_data = pd.DataFrame()
# this for loop does 3 things
            for z in range(len(rates2)-2):
            # checks that the players being evaluated are the same
                pitch_1 = rates2.pitch_type[z]
                pitch_2 = rates2.pitch_type[z+1]
                pitch_3 = rates2.pitch_type[z+2]
                name = rates2.playerid[z]
                last_year = rates2.game_year[z+2]
                year_2 = rates2.game_year[z]
                year = rates2.game_year[z+1]
                name_2 = rates2.playerid[z+1]
                name_3 = rates2.playerid[z+2]
                if name == name_2 == name_3 and (last_year - year_2) == 2 and pitch_1 == pitch_2 == pitch_3:
                    bbe_test = rates2.bip[z+2]
                    if bbe_test < pa_c:
                        continue
                    else:
                        pass
                else:
                    continue
            # if both true, takes their data and grabs random bbe from both "prior" years
                data_2 = pitchers.query('playerid == @name and game_year == @year_2 and pitch_type == @pitch_1').reset_index(drop=True)
                data_2['years_ago'] = 2
                data_1 = pitchers.query('playerid == @name and game_year == @year and pitch_type == @pitch_2').reset_index(drop=True)
                data_1['years_ago'] = 1
                if sum(data_2.in_play) < pa_two or sum(data_1.in_play) < pa_p:
                    continue
                else:
                    pass
                for data in [data_2,data_1]:
                    if data.years_ago[0] == 1:
                        samp = pa_p
                    else:
                        samp = pa_two
                    if samp == 0:
                        continue
                    else:
                        pass
                    if sum(data.in_play) == samp:
                        past_data = past_data.append(data)
                        continue
                    else:
                        pass
                    data_index = np.arange(len(data))
                    np.random.shuffle(data_index)
            # average pitches per bbe is 5.73
                    total = ceil(5.7*samp)
                    select_rows = data_index[:total]
                    subset = data.iloc[select_rows].copy().reset_index(drop=True)
                    add_on = 2
                    while sum(subset.in_play) != samp:
                        if sum(subset.in_play) > samp:
                            subset = subset.iloc[:-1]
                        elif sum(subset.in_play) < samp:
                            subset = subset.append(data.iloc[data_index[total+(add_on-1):total+add_on]])
                            add_on += 1
                    subset = subset.reset_index(drop=True)
                    if sum(subset.ground_ball) == 0:
                        subset = subset.drop(randint(0,len(subset)-1)).reset_index(drop=True)
                        gbd = data.query('ground_ball == 1').reset_index(drop=True)
                        row = randint(0,len(gbd)-1)
                        subset = subset.append(data.iloc[row]).reset_index(drop=True)
                    elif sum(subset.fly_ball) == 0:
                        subset = subset.drop(randint(0,len(subset)-1)).reset_index(drop=True)
                        gbd = data.query('fly_ball == 1').reset_index(drop=True)
                        row = randint(0,len(gbd)-1)
                        subset = subset.append(data.iloc[row]).reset_index(drop=True)
                    past_data = past_data.append(subset)
            past_data = past_data.reset_index(drop=True)
            past_data = past_data.drop_duplicates()
            
# since we grabbed the raw pitches, need to transform into rates again
            
            fbb = past_data.query('in_play ==1')
            fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
            past_grouped = past_data.groupby(['player_name','playerid', 'game_year', 'pitch_type','years_ago']).agg(
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
            x_move =('pfx_x','mean'),
            z_move =('pfx_z','mean'),
            extension =('release_extension','mean'),
            home_run =('home_run','sum'),
            pitch_count=('pitch_type','size')).reset_index()
            xgrouped = fbb.groupby(['player_name','playerid', 'game_year','pitch_type','years_ago']).agg(
            tot_wob =('estimated_woba_using_speedangle','sum'),
            max_ev=('launch_speed','max'),
            count=('player_name','size')).reset_index()
            xgrouped['xwobacon'] = round(xgrouped['tot_wob'].astype(float)/xgrouped['count'],3)
            xgrouped = xgrouped[['player_name','playerid','game_year','pitch_type','years_ago','xwobacon','max_ev']]
            past_grouped = past_grouped.merge(xgrouped,on=['player_name','playerid', 'game_year','pitch_type','years_ago'])
            past_grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid','pitch_type','years_ago', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(past_grouped.set_index(['player_name','playerid','pitch_type','years_ago','game_year']).index).values
            past_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']] = past_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']].astype(int)
            past_grouped['gb/fb'] = round(past_grouped['ground_ball']/past_grouped['fly_ball'],2)
            
            
            past_rates = past_grouped[['player_name','playerid','game_year','pitch_type','years_ago','xwobacon',
                 'la','ev','gb/fb','swing','pitch_count','bip','x_move','z_move','extension','velo','spin_rate']]
            past_rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
            past_rates['whiff'] = round(past_grouped['whiff']/past_grouped['swing'],3)*100
            past_rates['hh'] = round(past_grouped['hh']/past_grouped['bip'],3)*100
            past_rates['ld'] = round((past_grouped['line_drive'])/past_grouped['bip'],3)*100
            past_rates['hr'] = round((past_grouped['home_run'])/past_grouped['bip'],3)*100
            past_rates['barrel'] = round((past_grouped['barrels'])/past_grouped['bip'],3)*100
            past_rates['weak'] = round((past_grouped['poorly_hit'])/past_grouped['bip'],3)*100
            past_rates['swing%'] = round((past_grouped['swing'])/past_grouped['pitch_count'],3)*100
            past_rates['chase%'] = round((past_grouped['chase'])/past_grouped['swing'],3)*100
            past_rates = past_rates.drop(columns=['pitch_count','swing'])
            past_rates = past_rates.sort_values(by=['playerid','pitch_type','game_year'],ascending=True).reset_index(drop=True)
            
# similar process with current year data, less tricky b/c no random sample needed
            first_x = pd.DataFrame()
            after_100 = pd.DataFrame()
            names = past_rates[['player_name','playerid','pitch_type']].drop_duplicates().reset_index(drop=True)

            for x in range(len(names)):
                    name = names.playerid[x]
                    pitch = names.pitch_type[x]
                    subset = rates2.query('playerid == @name and pitch_type == @pitch').reset_index(drop=True)
                    if len(subset) < 3:
                        continue
                    for y in range(2,len(subset)):
                        bip = subset.bip[y]
                        bip_p = subset.bip[y-1]
                        bip_t = subset.bip[y-2]
                        year_2 = subset.game_year[y]
                        year_0 = subset.game_year[y-2]
                        if (year_2 - year_0) > 2:
                            continue
                        if bip >= pa_c and bip_p >= pa_p and bip_t >= pa_two:
                            if pa_c > 100:
                                year = subset.game_year[y]
                                check = pitchers.query('playerid == @name and game_year == @year and pitch_type == @pitch')
                                check = check.sort_values(by='game_date',ascending = True).reset_index(drop=True)
                                row = (check['in_play'].cumsum() == (pa_c-100)).idxmax()+1
                                prior = check.iloc[:row]                                
                                post = check.iloc[row:]
                                first_x = first_x.append(prior)
                                after_100 = after_100.append(post)
                            else:
                                year = subset.game_year[y]
                                check = pitchers.query('playerid == @name and game_year == @year and pitch_type == @pitch')
                                check = check.sort_values(by='game_date',ascending = True).reset_index(drop=True)
                                row = (check['in_play'].cumsum() == (100)).idxmax()+1
                                post = check.iloc[row:]
                                after_100 = after_100.append(post)
                        else:
                            continue
            if not first_x.empty:
                fbb = first_x.query('in_play ==1')
                fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
                first_grouped = first_x.groupby(['player_name','playerid', 'game_year', 'pitch_type']).agg(
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
                x_move =('pfx_x','mean'),
                z_move =('pfx_z','mean'),
                extension =('release_extension','mean'),
                home_run =('home_run','sum'),
                pitch_count=('pitch_type','size')).reset_index()
                xgrouped = fbb.groupby(['player_name','playerid', 'game_year','pitch_type']).agg(
                tot_wob =('estimated_woba_using_speedangle','sum'),
                max_ev=('launch_speed','max'),
                count=('player_name','size')).reset_index()
                xgrouped['xwobacon'] = round(xgrouped['tot_wob'].astype(float)/xgrouped['count'],3)
                xgrouped = xgrouped[['player_name','playerid','game_year','pitch_type','xwobacon','max_ev']]
                first_grouped = first_grouped.merge(xgrouped,on=['player_name','playerid', 'game_year','pitch_type'])
                first_grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid','pitch_type', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(first_grouped.set_index(['player_name','playerid','pitch_type','game_year']).index).values
                first_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']] = first_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']].astype(int)
                first_grouped['gb/fb'] = round(first_grouped['ground_ball']/first_grouped['fly_ball'],2)
                
                
                first_rates = first_grouped[['player_name','playerid','game_year','pitch_type','xwobacon',
                     'la','ev','gb/fb','swing','pitch_count','bip','x_move','z_move','extension','velo','spin_rate']]
                first_rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
                first_rates['whiff'] = round(first_grouped['whiff']/first_grouped['swing'],3)*100
                first_rates['hh'] = round(first_grouped['hh']/first_grouped['bip'],3)*100
                first_rates['ld'] = round((first_grouped['line_drive'])/first_grouped['bip'],3)*100
                first_rates['hr'] = round((first_grouped['home_run'])/first_grouped['bip'],3)*100
                first_rates['barrel'] = round((first_grouped['barrels'])/first_grouped['bip'],3)*100
                first_rates['weak'] = round((first_grouped['poorly_hit'])/first_grouped['bip'],3)*100
                first_rates['swing%'] = round((first_grouped['swing'])/first_grouped['pitch_count'],3)*100
                first_rates['chase%'] = round((first_grouped['chase'])/first_grouped['swing'],3)*100
                first_rates = first_rates.drop(columns=['pitch_count','swing','bip'])
                first_rates = first_rates.sort_values(by=['playerid','pitch_type','game_year'],ascending=True).reset_index(drop=True)

            else:
                pass
            fbb = after_100.query('in_play ==1')
            fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
            after_grouped = after_100.groupby(['player_name','playerid', 'game_year', 'pitch_type']).agg(
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
            x_move =('pfx_x','mean'),
            z_move =('pfx_z','mean'),
            extension =('release_extension','mean'),
            home_run =('home_run','sum'),
            pitch_count=('pitch_type','size')).reset_index()
            xgrouped = fbb.groupby(['player_name','playerid', 'game_year','pitch_type']).agg(
            tot_wob =('estimated_woba_using_speedangle','sum'),
            max_ev=('launch_speed','max'),
            count=('player_name','size')).reset_index()
            xgrouped['xwobacon'] = round(xgrouped['tot_wob'].astype(float)/xgrouped['count'],3)
            xgrouped = xgrouped[['player_name','playerid','game_year','pitch_type','xwobacon','max_ev']]
            after_grouped = after_grouped.merge(xgrouped,on=['player_name','playerid', 'game_year','pitch_type'])
            after_grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid','pitch_type', 'game_year'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(after_grouped.set_index(['player_name','playerid','pitch_type','game_year']).index).values
            after_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']] = after_grouped[['hh','barrels','poorly_hit','fly_ball','ground_ball','line_drive','whiff','chase','swing','home_run','bip']].astype(int)
            after_grouped['gb/fb'] = round(after_grouped['ground_ball']/after_grouped['fly_ball'],2)
            
            
            after_rates = after_grouped[['player_name','playerid','game_year','pitch_type','xwobacon',
                 'la','ev','gb/fb','swing','pitch_count','bip','x_move','z_move','extension','velo','spin_rate']]
            after_rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
            after_rates['whiff'] = round(after_grouped['whiff']/after_grouped['swing'],3)*100
            after_rates['hh'] = round(after_grouped['hh']/after_grouped['bip'],3)*100
            after_rates['ld'] = round((after_grouped['line_drive'])/after_grouped['bip'],3)*100
            after_rates['hr'] = round((after_grouped['home_run'])/after_grouped['bip'],3)*100
            after_rates['barrel'] = round((after_grouped['barrels'])/after_grouped['bip'],3)*100
            after_rates['weak'] = round((after_grouped['poorly_hit'])/after_grouped['bip'],3)*100
            after_rates['swing%'] = round((after_grouped['swing'])/after_grouped['pitch_count'],3)*100
            after_rates['chase%'] = round((after_grouped['chase'])/after_grouped['swing'],3)*100
            after_rates = after_rates.drop(columns=['pitch_count','swing'])
            after_rates = after_rates.sort_values(by=['playerid','pitch_type','game_year'],ascending=True).reset_index(drop=True)
            after_rates = after_rates.drop(columns='bip')
            past_rates = past_rates.drop(columns='bip')
            
            for i in range(4,after_rates.shape[1]):
                if first_x.empty:
                    column = after_rates.columns[i]
                    merge = after_rates[['player_name','playerid','game_year','pitch_type',column]].sort_values(by=['playerid','pitch_type','game_year'],ascending=False).reset_index(drop=True)
                    column_b = column+'_x'
                    column_a = column+'_y'
                    column_c = column+'_2'
                else:
                    column = past_rates.columns[i+1]
                    first = first_rates[['player_name','playerid','game_year','pitch_type',column]]
                    after = after_rates[['player_name','playerid','game_year','pitch_type',column]]
                    merge = first.merge(after,how='inner',on=['playerid','game_year','pitch_type']).sort_values(by=['playerid','game_year','pitch_type'],ascending=False).reset_index(drop=True)
                    column_b = column+'_x'
                    column_a = column+'_y'
                    column_c = column+'_2'
                for z in range(len(merge)):
                    year1 = (merge.game_year[z])-1
                    year2 = (merge.game_year[z])-2
                    name = merge.playerid[z]
                    pitch = merge.pitch_type[z]
                    data = past_rates.query('playerid == @name and ((game_year == @year1 and years_ago == 1) or (game_year == @year2 and years_ago == 2)) and pitch_type == @pitch')[['player_name','playerid','game_year','pitch_type',column]]
                    if z == 0:
                        past = data
                    else:
                        past = past.append(data)
                past = past.sort_values(by=['playerid','pitch_type','game_year'],ascending=False).reset_index(drop=True)
                evens = past.iloc[::2].reset_index(drop=True)
                odds = past.iloc[1::2].reset_index(drop=True)
                evens[f'{column_c}'] = odds[column]
                evens.game_year = evens.game_year + 1
                merge = merge.merge(evens,how='inner',on=['playerid','game_year','pitch_type']).sort_values(by=['playerid','game_year','pitch_type'],ascending=False).reset_index(drop=True)
                gb_check = merge[merge.isin([inf,nan]).any(axis=1)]
                if gb_check.empty:
                    pass
                else:
                    merge = merge.drop(gb_check.index).reset_index(drop=True)
                if not first_x.empty:
                    def objective(weights):
                        predictions = weights[0] * merge[column_b] + weights[1] * merge[column] + weights[2] * merge[column_c]
                        return -pearsonr(predictions, merge[column_a])[0]
                    
                    result = minimize(objective, x0=[0.33, 0.33, 0.34], 
                                      bounds=[(0, 1), (0, 1), (0, 1)],
                                      constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
                    
                    best_weights = result.x.round(3)
                    if i == 4:
                        results = pd.DataFrame([[column,best_weights[0],best_weights[1],best_weights[2],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])
                    else:
                        results = pd.concat([results,pd.DataFrame([[column,best_weights[0],best_weights[1],best_weights[2],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])],ignore_index=True)
            
                else:
                    def objective(weights):
                        predictions = weights[0] * merge[column_a] + weights[1] * merge[column_c]
                        return -pearsonr(predictions, merge[column_b])[0]
                    
                    result = minimize(objective, x0=[.5,.5], 
                                      bounds=[(0, 1), (0, 1)],
                                      constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
                    
                    best_weights = result.x.round(2)
                    if i == 3:
                        results = pd.DataFrame([[column,0,best_weights[0],best_weights[1],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])
                    else:
                        results = pd.concat([results,pd.DataFrame([[column,0,best_weights[0],best_weights[1],round(-result.fun,2) ]], columns=['stat', 'pre_split','prior_year','two_years','correlation'])],ignore_index=True)
            results[['pre_split_bbe','prior_bbe','two_years_bbe','data_points']] = (pa_c)-100,pa_p,pa_two,len(merge)
            final_results = final_results.append(results)
            final_results.to_csv('bbe_weights_pitchers.csv',index=False)
