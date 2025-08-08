# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:54:09 2025

@author: Brendan
"""
#%%
def modify_batters(batters,old_hits,players,ids):
    import pandas as pd
    from numpy import select,nan,inf
    from unidecode import unidecode
    import numpy as np
    from statistics import mean
    from math import floor, ceil
    
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
        
    player_sums= pd.DataFrame((grouped.groupby(['player_name','playerid']).agg({
        **{col: 'sum' for col in list(grouped.columns[4:16])}})).reset_index())
    
    player_sums[['xwobacon','max_ev','ev','la']] = 0
    for i in range(len(player_sums)):
        name = player_sums.playerid[i]
        for col in player_sums.columns[2:]:
            if col in ['pitch_count','bip']:
                continue
            elif col in ['xwobacon']:
                player_sums[col][i] = round(mean(fbb.query('playerid==@name').estimated_woba_using_speedangle),3)
            elif col in ['ev']:
                player_sums[col][i] = round(mean(fbb.query('playerid==@name').launch_speed),3)
            elif col in ['la']:
                player_sums[col][i] = round(mean(fbb.query('playerid==@name').launch_angle),3)
            elif col == 'max_ev':
                player_sums[col][i] = max(xgrouped.query('playerid==@name').max_ev)
            else:
                if col == 'barrels':
                    o_col = 'barrel'
                elif col == 'poorly_hit':
                    o_col = 'weak'
                else:
                    o_col = col
                player_sums[col][i] = round(sum(batters.query('playerid==@name')[o_col])/len(fbb.query('playerid==@name')),3)
                
    
    for i in range(len(grouped)):
        name = grouped.playerid[i]
        if sum(player_sums.query('playerid == @name').bip) < 30:
            if grouped.bip[i] < 30:
                diff = 30 - grouped.bip[i]
                added_sums = round(league_sums.iloc[:,:10] * diff,2)
                grouped.iloc[i:i+1,4:14] += (added_sums.iloc[:,:10]).values
                grouped.iloc[i:i+1,17:21] = (grouped.iloc[i:i+1,17:21]*((30-diff)/30))+((league_sums.iloc[0:1,12:].values)*(diff/30))
            else:
                continue
        else:
            if grouped.bip[i] < 30:
                diff = 30 - grouped.bip[i]
                p_weight = ceil(diff/2)
                l_weight = floor(diff/2)
                added_sums = round((league_sums.iloc[:,:10] * l_weight)+(league_sums.iloc[:,:10] * p_weight),2)
                grouped.iloc[i:i+1,4:14] += (added_sums.iloc[:,:10]).values
                grouped.iloc[i:i+1,17:21] = (grouped.iloc[i:i+1,17:21]*((30-diff)/30))+((league_sums.iloc[0:1,12:].values)*(l_weight/30))+((player_sums.iloc[0:1,14:].values)*(p_weight/30))
            else:
                continue
    # now that we have per-pitch averages for the league, we will regress for players with under 30 bip/30 pitches seen

    
    grouped = grouped.drop(grouped.query('fly_ball == 0').index).reset_index(drop=True)
    grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])
    rates = grouped[['player_name','playerid','game_year','pitch_type','xwobacon','la','ev','max_ev','gb/fb',
                     'pitch_count','bip','swing']]
    rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
    
    
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
            rates['chase%'][i] = round((grouped['chase'][i])/bip,3)*100
        else:    
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
            player.iloc[:,3:16] = (player.iloc[:,3:16]*(0.4 + (0.35*p_weight)) + (1*(1-(0.4 + (0.35*p_weight)))))
        else:
            player.iloc[:,3:16] = (player.iloc[:,3:16]*0.75) + 0.25
        player = round(player,2)
        weighted = weighted.append(player)
        weighted = weighted.reset_index(drop=True)
        player = round(player,2)
        weighted = weighted.append(player)
    weighted = weighted.drop_duplicates().reset_index(drop=True)
    weighted = weighted.drop(columns=['game_year','hr'])
    weighted['pred_hr'] = result.predict(weighted[old_hits.iloc[:,3:16].drop(columns='hr').columns]).round(2)
    
    
    
    
        
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
            if sum(pitch_2025.pitch_count) >= 200:
                for col in pitch.columns[20:]:
                   pitch[col] = pitch_2025[col]
            else:
                for col in pitch.columns[20:]:
                    pitch[col] = pitch_2024[col].sum() + pitch_2025[col].sum() if len(pitch_2024) > 0 and len(pitch_2025) > 0 else pitch[col]
        else:
            pass
        per_pitch_short = per_pitch_short.append(pitch)
    per_pitch_short = per_pitch_short.reset_index(drop=True)
    per_pitch_short['pred_hr'] = result.predict(per_pitch_short[old_hits.iloc[:,3:16].drop(columns='hr').columns]).round(2)
    per_pitch_short = per_pitch_short.drop(columns='game_year')
    
    
    """Now that we have all players stats, we must get their splits based on
    the handedness of the pitcher they're facing so that can be factored in
    
    pretty much doing the exact same process but breaking up by handedness
    instead of pitch_type"""
    
    grouped = batters.groupby(['player_name','playerid', 'stand','p_throws']).agg(
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
    
    
    xgrouped = fbb.groupby(['player_name','playerid', 'stand','p_throws']).agg(
        tot_wob =('estimated_woba_using_speedangle','sum'),
        max_ev=('launch_speed','max'),
    count=('pitch_type','size')).reset_index()
    xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
    xgrouped = xgrouped[['player_name','playerid','stand','p_throws','xwobacon','max_ev']]
    grouped = grouped.merge(xgrouped,on=['player_name','playerid','stand','p_throws'])
    grouped[['ev', 'la']] = fbb.groupby(['player_name','playerid', 'stand','p_throws' ])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name','playerid', 'stand','p_throws']).index).values
    
    
    # getting averages for every pitch
    
    pitch_list = batters[['stand','p_throws']]
    pitch_list = pitch_list.drop_duplicates().reset_index(drop=True)
    pitch_avgs = grouped.groupby(['stand','p_throws']).agg({**{col: 'sum' for col in list(grouped.columns[4:])}}).reset_index()
    pitch_avgs[['xwobacon','max_ev','ev','la']] = 0
    from statistics import mean
    for i in range(len(pitch_avgs)):
        stand = pitch_avgs.stand[i]
        throws = pitch_avgs.p_throws[i]
        subset = grouped.query('stand == @stand and p_throws == @throws').reset_index(drop=True)
        for col in pitch_avgs.columns[15:]:
            total = 0
            for p in range(len(subset)):
                if col == 'max_ev':
                    total = mean(subset.max_ev)
                else:
                    value = subset[col][p]*(subset.pitch_count[p]/sum(subset.pitch_count)).round(5)
                    total += value
            total = round(total,3)
            pitch_avgs[col][i] = total
    pitch_avgs['gb/fb'] = (pitch_avgs['ground_ball']/pitch_avgs['fly_ball']).round(2)
    pitch_avgs = pitch_avgs.drop(columns=['fly_ball','ground_ball'])
    
    for col in pitch_avgs.columns[2:10]:
        if col in ['chase','whiff']:
            pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['swing'])*100,2)
        elif col == 'swing':
            pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['pitch_count'])*100,2)
        elif col=='home_run':
            pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['bip'])*100,4)
        else:
            pitch_avgs[col] = round((pitch_avgs[col]/pitch_avgs['bip'])*100,2)
    pitch_avgs = pitch_avgs.reset_index(drop=True)
    
    
    # getting league averages to regress small sample sizes
    
    league_sums= pd.DataFrame((grouped.groupby(['stand','p_throws']).agg({
        **{col: 'sum' for col in list(grouped.columns[4:])}})).sum().reset_index()).T
    league_sums.columns = league_sums.loc['index']
    league_sums = league_sums.reset_index(drop=True).drop(0)
    
    league_sums[['xwobacon','max_ev','ev','la']] = 0
    for col in league_sums.columns:
        if col in ['pitch_count','max_ev','bip','age']:
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
                grouped.iloc[i:i+1,17:22] = (grouped.iloc[i:i+1,17:22]*(grouped.bip[i]/30))+((league_sums.iloc[0:1,13:].values)*(diff/30))
                grouped.pitch_count[i] += round(league_sums.pitch_count[1]*diff)
                grouped.bip[i] = 30
        else:
            continue
    import numpy as np
    
    grouped = grouped.drop(grouped.query('fly_ball == 0').index).reset_index(drop=True)
    grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])
    rates = grouped[['player_name','playerid','stand','p_throws','xwobacon','la','ev','max_ev','gb/fb',
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
    
    rates = rates.reindex(columns=['player_name','playerid',
                                             'stand','p_throws','hh','barrel','weak',
                                             'ld','whiff','chase%',
                                             'swing%','hr',
                                             'xwobacon','max_ev','ev','la','bip','pitch_count','gb/fb'])
    
    # get averages for the league environment
    
    just_stats = rates.iloc[:,4:rates.shape[1]].drop(columns=['pitch_count','bip'])
    pitch_avgs = pitch_avgs.rename(columns = {'barrels':'barrel','chase':'chase%','home_run':'hr','line_drive':'ld',
                                              'poorly_hit':'weak','swing':'swing%'})
    for i in range(0,len(just_stats)):
        stand = rates.stand[i]
        throws = rates.p_throws[i]
        bucket_subset = pitch_avgs.query('stand == @stand and p_throws == @throws').reset_index(drop=True)
        if i == 0:
            new_stats = round(just_stats.loc[i]/(bucket_subset.iloc[:,2:bucket_subset.shape[1]].drop(columns=['pitch_count','bip','age'])),2)
        else:
            data = round(just_stats.loc[i]/(bucket_subset.iloc[:,2:bucket_subset.shape[1]].drop(columns=['pitch_count','bip','age'])),2)
            new_stats = new_stats.append(data)
    new_stats = new_stats.reset_index(drop=True)
    new_stats[['player_name','playerid','stand','p_throws','pitch_count','bip']] = rates[['player_name','playerid','stand','p_throws','pitch_count','bip']]
    new_stats["player_name"] = [" ".join(n.split(", ")[::-1]) for n in new_stats["player_name"]]
    new_stats['player_name'] = new_stats['player_name'].apply(unidecode)
    new_stats.iloc[:,:12] = new_stats.iloc[:,:12].astype(float).round(2)
        
    per_pitch_split = pd.DataFrame()
    options = new_stats[['player_name','playerid','stand','p_throws','bip']]
    options = options.drop_duplicates(subset=['player_name','playerid','p_throws','stand'], keep='first').reset_index(drop=True)
    new_stats = new_stats.drop(columns='hr')
    
    for i in range(0,len(options)):
        bbe = options['bip'][i]
        name = options['playerid'][i]
        throws = options['p_throws'][i]
        pitch = new_stats.query('playerid == @name and p_throws == @throws').reset_index(drop=True)
        player = weighted.query('playerid == @name').reset_index(drop=True)
        if bbe < 50:
            base_weight = (50 - bbe) / 50
            player_confidence = min(player.bip[0] / 200, 1.0)
            player_weight = base_weight * (0.5 + 0.4 * player_confidence)
            league_weight = base_weight - player_weight
            for p in range(0,11):
                pitch.iloc[:,p] = (pitch.iloc[:,p][0]*(bbe/30) + player.iloc[:,p+2][0]*(player_weight) + 1*(league_weight)).round(2)
        per_pitch_split = per_pitch_split.append(pitch)
        
    per_pitch_split = per_pitch_split.reset_index(drop=True)
    per_pitch_split['splits'] = per_pitch_split.apply(lambda row: 'plat_disadv' if row['stand'] == row['p_throws'] else 'plat_adv', axis=1)
    per_pitch_split = per_pitch_split.drop_duplicates()
    
    names['plat_disc'] = 0
    for i in range(0,len(names)):
        p_id = names['playerid'][i]
        player = per_pitch_split.query('playerid == @p_id')
        dis = player.query('splits == "plat_disadv"').reset_index(drop=True)
        adv = player.query('splits == "plat_adv"').reset_index(drop=True)
        if len(adv) != 1 or len(dis) != 1:
            continue
        else:
            pass
        plat_diff = dis.iloc[:,:12]-adv.iloc[:,:12]
        plat_diff['pred_diff'] = result.predict(plat_diff).round(2)
        names['plat_disc'][i] = plat_diff['pred_diff'][0]
        
    per_pitch_short = per_pitch_short.merge(names,on=['player_name','playerid'])

    players['team_id'] = players['team_id'].astype(int)
    players = players.merge(ids, on='team_id', how='left')
    players = players[['person_id','Stadium','person_full_name']]
    players = players.rename(columns={'person_id':'playerid','person_full_name':'player_name'})
    players['playerid'] = players['playerid'].astype(int)
    per_pitch_short = per_pitch_short.merge(players,how='inner',on=['playerid','player_name'])
    per_pitch_short = per_pitch_short.drop_duplicates()
    
    per_pitch_short['player_name'] = per_pitch_short['player_name'].apply(unidecode)
    per_pitch_short = per_pitch_short.reset_index(drop=True)
    
    return per_pitch_short
