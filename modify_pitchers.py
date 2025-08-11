# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:47:42 2025

@author: Brendan
"""

def modify_pitchers(pitchers, old_pitch,players,ids):
    import pandas as pd
    from numpy import select,nan,inf
    from unidecode import unidecode
    import numpy as np
    from statistics import mean
    from math import floor, ceil
    
    fbb = pitchers.query('in_play==1')
    fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
    
    # getting average bf and outs gotten for every starter
    
    outings = pitchers.groupby(['player_name','playerid','batter','game_date','inning','outs_when_up','events']).agg(
        pitches=('batter','size'))
    outings = outings.reset_index()
    outings['outs'] = ((outings['inning']*3)-3) + outings['outs_when_up']
    outings = outings.groupby(['player_name','playerid','game_date']).agg(
        bf=('game_date', 'size'),
        last_out=('outs','max'),
        first_out=('outs','min'))
    outings = outings.reset_index()
    outings['outs'] = outings.last_out - outings.first_out
    outings = outings.drop(columns=['first_out','last_out'])
    pitch_counts = pitchers.groupby(['playerid', 'game_date']).agg(pitch_count=('pitch_type','size'))
    outings = outings.merge(pitch_counts, on=['playerid', 'game_date'], how='left')
    
    last_events = pitchers.groupby(['playerid', 'game_date'])['events'].last().reset_index(name='last_event')
    outings = outings.merge(last_events, on=['playerid', 'game_date'], how='left')
    
    outings['outs'] += outings['last_event'].str.contains('triple_play', na=False).astype(int) * 3
    outings['outs'] += outings['last_event'].str.contains('double_play', na=False).astype(int) * 2
    outings['outs'] += (outings['last_event'].str.contains('out', na=False) | outings['last_event'].str.contains('sac', na=False)).astype(int) * 1
    outings = outings.drop(columns=['last_event'])
    
    outings = outings.groupby(['player_name','playerid']).agg(
        avg_bf=('bf','mean'),
        std_bf=('bf','std'),
        avg_outs=('outs','mean'),
        atd_outs=('outs','std'),
        avg_pc=('pitch_count','mean'),
        std_pc=('pitch_count','std'),
        apps=('playerid','size')).reset_index()
    outings = outings.round(1)
    outings["player_name"] = [" ".join(n.split(", ")[::-1]) for n in outings["player_name"]]
    
    
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
            league_sums[col][1] = round(sum(pitchers.in_play)/len(pitchers),3)
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

    grouped = grouped.drop(grouped.query('fly_ball == 0').index).reset_index(drop=True)
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
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LassoCV
    import numpy as np
    old_pitch = pd.read_csv('old_pitch.csv')
    X = old_pitch.iloc[:,3:20].drop(columns='hr')
    y = old_pitch.hr
    result = LassoCV(alphas=np.logspace(-3, 1, 20),cv=5, random_state=14, max_iter=10000)
    result = result.fit(X, y)  
    
    
    
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
        if len(player['year'].unique()) == 2:
            player = player.groupby(['player_name','playerid']).agg({
                **{col: 'mean' for col in list(player.columns[3:20])},
                **{col: 'sum' for col in player.columns[20:]}}).reset_index()
        elif sum(player.bip) < 200 and sum(player.year) == 2025:
            p_weight = round(sum(player.bip)/200,2)
            player.iloc[:,3:20] = (player.iloc[:,3:20]*(0.4 + (0.35*p_weight)) + (1*(1-(0.4 + (0.35*p_weight)))))
        else:
            player.iloc[:,3:20] = (player.iloc[:,3:20]*0.75) + 0.25
        player = round(player,2)
        weighted = weighted.append(player)
    weighted = weighted.drop_duplicates().reset_index(drop=True)
    weighted = weighted.drop(columns=['year','hr'])
    weighted['pred_hr'] = result.predict(weighted[old_pitch.iloc[:,3:20].drop(columns='hr').columns]).round(2)
    
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
        player = weighted.query('playerid == @name').reset_index(drop=True)
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
            pass
        per_pitch_short = per_pitch_short.append(pitch)
    per_pitch_short = per_pitch_short.reset_index(drop=True)
    per_pitch_short['pred_hr'] = result.predict(per_pitch_short[old_pitch.iloc[:,3:20].drop(columns='hr').columns]).round(2)
    per_pitch_short = per_pitch_short.drop(columns='year')
    
    players['team_id'] = players['team_id'].astype(int)
    per_pitch_short = per_pitch_short.merge(outings,how='inner',on=['playerid','player_name'])
    players = players.merge(ids, on='team_id', how='left')
    players = players[['person_id','Stadium','person_full_name']]
    players = players.rename(columns={'person_id':'playerid','person_full_name':'player_name'})
    players['playerid'] = players['playerid'].astype(int)
    players = players.drop_duplicates().reset_index(drop=True)

    per_pitch_short['player_name'] = per_pitch_short['player_name'].apply(unidecode)
    per_pitch_short = per_pitch_short.merge(players,how='inner',on=['playerid','player_name'])
    per_pitch_short = per_pitch_short.drop_duplicates()
    per_pitch_short = per_pitch_short.reset_index(drop=True)
    
    """since we cannot aggregate things like velo, movement, and spin rate,
    as a pitcher's goal is to vary those things, we will not be getting splits
    for pitchers"""
    
    
    
    return per_pitch_short
