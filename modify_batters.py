# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:54:09 2025

@author: Brendan
"""
#%%
def modify_batters(batters,old_hits,date):
    import pandas as pd
    from numpy import select,nan,inf
    from unidecode import unidecode
    import numpy as np
    from statistics import mean
    from math import floor, ceil
    bbe = pd.read_csv('bbe_weights.csv')
    data_year = int(date[:4])-2
    c_year = int(date[:4])
    batters = batters.query('game_year >= @data_year and game_date <= @date')
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
    """
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
    """
    
    grouped = grouped.drop(grouped.query('fly_ball == 0').index).reset_index(drop=True)
    grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])
    rates = grouped[['player_name','playerid','game_year','pitch_type','xwobacon','la','ev','max_ev','gb/fb',
                     'pitch_count','bip','swing']]
    rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
    rates['whiff'] = round(grouped['whiff']/grouped['swing'],3)*100
    rates['hh'] = round(grouped['hh']/grouped['bip'],3)*100
    rates['ld'] = round((grouped['line_drive'])/grouped['bip'],3)*100
    rates['hr'] = round((grouped['home_run'])/grouped['bip'],3)*100
    rates['barrel'] = round((grouped['barrels'])/grouped['bip'],3)*100
    rates['weak'] = round((grouped['poorly_hit'])/grouped['bip'],3)*100
    rates['swing%'] = round((grouped['swing'])/grouped['pitch_count'],3)*100
    rates['chase%'] = round((grouped['chase'])/grouped['swing'],3)*100
            
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
    
    per_pitch_short = pd.DataFrame()
    thresholds = np.array([0,10, 25, 50, 75, 100,150,200])
    for i in range(len(names)):
        name = names['playerid'][i]
        player = new_stats.query('playerid == @name').reset_index().drop(columns='index')
        pitch_list = list(player.pitch_type.unique())
        for pitch in pitch_list:
            subset = player.query('pitch_type == @pitch').sort_values(by='game_year',ascending=False).reset_index(drop=True)
            subset['thresholds'] = 0
            if subset.loc[0].bip < 10:
                subset = subset.drop(0).reset_index(drop=True)
            for q in range(len(subset)):
                    samp = subset.bip[q]
                    subset['thresholds'][q] = thresholds[np.abs(thresholds - samp).argmin()]
                    if subset['thresholds'][q] == 0 and subset['game_year'][q] != c_year:
                        subset = subset.drop(q)
            subset = subset.reset_index(drop=True)
            if len(subset) == 3:
                tya = subset.thresholds[2]
                pys = subset.thresholds[1]
                psb = subset.thresholds[0]
                bbe_subset = bbe.query('pre_split_bbe == @psb and prior_bbe == @pys and two_years_bbe == @tya')
                bbe_mods = bbe_subset.iloc[:,:4].T.reset_index(drop=True)
                bbe_mods.columns = bbe_mods.iloc[0]
                bbe_mods = bbe_mods.drop(0).reset_index(drop=True)
                subset.iloc[:,:13] = subset.iloc[:,:13].values*bbe_mods.values
                stats = pd.DataFrame(round(subset.iloc[:,:13].sum(),2)).T
            elif len(subset) == 2 and max(subset.game_year) != c_year:
                tya = subset.thresholds[1]
                pys = subset.thresholds[0]
                bbe_subset = bbe.query('pre_split_bbe == 0 and prior_bbe == @pys and two_years_bbe == @tya')
                bbe_mods = bbe_subset.iloc[:,:4].T.reset_index(drop=True)
                bbe_mods.columns = bbe_mods.iloc[0]
                bbe_mods = bbe_mods.drop([0,1]).reset_index(drop=True)
                subset.iloc[:,:13] = subset.iloc[:,:13].values*bbe_mods.values
                stats = pd.DataFrame(round(subset.iloc[:,:13].sum(),2)).T
            elif len(subset) == 2 and max(subset.game_year) == c_year:
                if min(subset.game_year) == data_year:
                    tya = subset.thresholds[1]
                    psb = subset.thresholds[0]
                    bbe_subset = bbe.query('pre_split_bbe == @psb and two_years_bbe == @tya')
                    bbe_subset = bbe_subset.groupby('stat').agg(
                        ps=('pre_split','mean'),
                        py=('prior_year','mean'),
                        ty=('two_years','mean'))
                    bbe_subset['ps_share'] = round(bbe_subset.ps/(bbe_subset.ps+bbe_subset.ty),2)
                    bbe_subset['ty_share'] = 1-bbe_subset.ps_share
                    bbe_subset.ps = bbe_subset.ps + (bbe_subset.py * bbe_subset.ps_share)
                    bbe_subset.ty = bbe_subset.ty + (bbe_subset.py * bbe_subset.ty_share)
                    bbe_subset = bbe_subset.drop(columns=['py','ty_share','ps_share'])
                    bbe_mods = bbe_subset.iloc[:,:4].T.reset_index(drop=True)
                    subset.iloc[:,:13] = subset.iloc[:,:13].values*bbe_mods.values
                    stats = pd.DataFrame(round(subset.iloc[:,:13].sum(),2)).T
                if min(subset.game_year) == (data_year+1):
                    tya = subset.thresholds[1]
                    psb = subset.thresholds[0]
                    bbe_subset = bbe.query('pre_split_bbe == @psb and prior_bbe == @tya')
                    bbe_subset = bbe_subset.groupby('stat').agg(
                        ps=('pre_split','mean'),
                        py=('prior_year','mean'),
                        ty=('two_years','mean'))
                    bbe_subset['ps_share'] = round(bbe_subset.ps/(bbe_subset.ps+bbe_subset.py),2)
                    bbe_subset['ty_share'] = 1-bbe_subset.ps_share
                    bbe_subset.ps = bbe_subset.ps + (bbe_subset.ty * bbe_subset.ps_share)
                    bbe_subset.py = bbe_subset.py + (bbe_subset.ty * bbe_subset.ty_share)
                    bbe_subset = bbe_subset.drop(columns=['ty','ty_share','ps_share'])
                    bbe_mods = bbe_subset.iloc[:,:4].T.reset_index(drop=True)
                    subset.iloc[:,:13] = subset.iloc[:,:13].values*bbe_mods.values
                    stats = pd.DataFrame(round(subset.iloc[:,:13].sum(),2)).T
            elif len(subset) == 1:
                if subset.game_year[0] == c_year:
                    bbes = subset.thresholds[0]
                    bbe_subset = bbe.query('pre_split_bbe == @bbes')
                    weight = round(mean(bbe_subset.pre_split),4)
                elif subset.game_year[0] == (data_year+1):
                    bbes = subset.thresholds[0]
                    bbe_subset = bbe.query('prior_bbe == @bbes')
                    weight = round(mean(bbe_subset.prior_year),4)
                else:
                    bbes = subset.thresholds[0]
                    bbe_subset = bbe.query('two_years_bbe == @bbes')
                    weight = round(mean(bbe_subset.two_years),4)
                avg_w = 1-weight
                subset.iloc[:,:13] = ((subset.iloc[:,:13].values*weight)+(avg_w)).round(2)
                stats = pd.DataFrame(round(subset.iloc[:,:13].sum(),2)).T
            if subset.empty:
                continue
            stats[['player_name','playerid','pitch_type','bip']] = subset[['player_name','playerid','pitch_type','bip']].loc[0].values
            per_pitch_short = per_pitch_short.append(stats)
                
                
    per_pitch_short = per_pitch_short.reset_index(drop=True)
    per_pitch_short['pred_hr'] = result.predict(per_pitch_short[old_hits.iloc[:,3:16].drop(columns='hr').columns]).round(2)    
    
    
    

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
    
    """
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
        """
    import numpy as np
    
    grouped = grouped.drop(grouped.query('fly_ball == 0').index).reset_index(drop=True)
    grouped['gb/fb'] = (grouped['ground_ball']/grouped['fly_ball'])
    rates = grouped[['player_name','playerid','stand','p_throws','xwobacon','la','ev','max_ev','gb/fb',
                     'pitch_count','bip','swing']]
    rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
    #split_metrics[[averages.columns[[range(2,len(averages.columns))]]]] = 0
    rates[['whiff','hh','ld','barrel','weak','hr','swing%','chase%']] = 0
    rates['whiff'] = round(grouped['whiff']/grouped['swing'],3)*100
    rates['hh'] = round(grouped['hh']/grouped['bip'],3)*100
    rates['ld'] = round((grouped['line_drive'])/grouped['bip'],3)*100
    rates['hr'] = round((grouped['home_run'])/grouped['bip'],3)*100
    rates['barrel'] = round((grouped['barrels'])/grouped['bip'],3)*100
    rates['weak'] = round((grouped['poorly_hit'])/grouped['bip'],3)*100
    rates['swing%'] = round((grouped['swing'])/grouped['pitch_count'],3)*100
    rates['chase%'] = round((grouped['chase'])/grouped['swing'],3)*100
    
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
    options = new_stats[['player_name','playerid','stand','p_throws','bip']].sort_values(by='player_name')
    options = options.drop_duplicates(subset=['player_name','playerid','p_throws','stand'], keep='first').reset_index(drop=True)
    new_stats = new_stats.drop(columns='hr')
    
    for i in range(0,len(options)):
        bip = options['bip'][i]
        name = options['playerid'][i]
        throws = options['p_throws'][i]
        pitch = new_stats.query('playerid == @name and p_throws == @throws').reset_index(drop=True)
        player = per_pitch_short.query('playerid == @name').reset_index(drop=True)
        if bip < 50:
            base_weight = (50 - bip) / 50
            play_stats = pd.DataFrame(player.iloc[:,:13].sum()/len(player)).T
            play_stats['bip'] = sum(player.bip)
            player_confidence = min(play_stats.bip[0] / 200, 1.0)
            player_weight = base_weight * (0.5 + 0.4 * player_confidence)
            league_weight = base_weight - player_weight
            for p in range(0,11):
                pitch.iloc[:,p] = (pitch.iloc[:,p][0]*(bip/50) + (play_stats.iloc[:,p+2][0]*(player_weight) + (league_weight))).round(2)
        per_pitch_split = per_pitch_split.append(pitch)
        
    per_pitch_split = per_pitch_split.reset_index(drop=True)
    per_pitch_split['splits'] = per_pitch_split.apply(lambda row: 'plat_disadv' if row['stand'] == row['p_throws'] else 'plat_adv', axis=1)
    per_pitch_split = per_pitch_split.drop_duplicates().dropna()
    
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
        plat_diff['pred_diff'] = result.predict(plat_diff[old_hits.iloc[:,3:16].drop(columns='hr').columns]).round(2)
        names['plat_disc'][i] = plat_diff['pred_diff'][0]
    
    per_pitch_short = per_pitch_short.merge(names,on=['player_name','playerid'])

    
    return per_pitch_short
