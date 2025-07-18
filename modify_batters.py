# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:54:09 2025

@author: Brendan
"""
#%%
def modify_batters(batters,old_hits,players,ids):
    import pandas as pd
    from numpy import select,nan
    from unidecode import unidecode
    import numpy as np
    
    fbb = batters.query('bbe==1')
    fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])

    grouped = batters.groupby(['player_name','hitter', 'year', 'pitch_type']).agg(
        hh=('hh', 'sum'), # Count of batted balls with exit velocity over 95 mph
        bbe=('bbe','sum'),
        barrels =('barrel','sum'),
        poorly_hit=('weak','sum'),
        fly_ball =('fly_ball','sum'),
        ground_ball =('ground_ball','sum'),
        line_drive =('line_drive','sum'),
        whiff =('whiff','sum'),
        swing =('swing','sum'),
        home_run =('home_run','sum'),
        pitch_count=('pitch_type', 'size')
        ).reset_index()
    xgrouped = fbb.groupby(['player_name','hitter', 'year', 'pitch_type']).agg(
        tot_wob =('estimated_woba_using_speedangle','sum'),
        count=('pitch_type','size')).reset_index()
    xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
    xgrouped = xgrouped[['player_name','hitter','year','pitch_type','xwobacon']]
    grouped = grouped.merge(xgrouped,on=['player_name','hitter','year','pitch_type'])

    grouped[['ev', 'la']] = batters.groupby(['player_name', 'hitter', 'year', 'pitch_type'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name', 'hitter', 'year', 'pitch_type']).index).values
    
    # turn counting stats into rate stats

    rates = grouped[['player_name','hitter','year','pitch_type','xwobacon','la','ev']]
    rates[['whiff','hh','fb','gb','ld','barrel','weak','hr','swing%','count','bbe']] = 0
    #split_metrics[[averages.columns[[range(2,len(averages.columns))]]]] = 0
    for i in range(0,len(grouped)):
        rates['whiff'][i] = round(grouped['whiff'][i]/grouped['swing'][i],3)*100
        rates['hh'][i] = round(grouped['hh'][i]/grouped['bbe'][i],3)*100
        rates['gb'][i] = round((grouped['ground_ball'][i])/grouped['bbe'][i],3)*100
        rates['fb'][i] = round((grouped['fly_ball'][i])/grouped['bbe'][i],3)*100
        rates['ld'][i] = round((grouped['line_drive'][i])/grouped['bbe'][i],3)*100
        rates['hr'][i] = round((grouped['home_run'][i])/grouped['bbe'][i],3)*100
        rates['barrel'][i] = round((grouped['barrels'][i])/grouped['bbe'][i],3)*100
        rates['weak'][i] = round((grouped['poorly_hit'][i])/grouped['bbe'][i],3)*100
        rates['swing%'][i] = round((grouped['swing'][i])/grouped['pitch_count'][i],3)*100
        rates['count'][i] = grouped['pitch_count'][i]
        rates['bbe'][i] = grouped['bbe'][i]

    # getting averages for every pitch
    pitch_list = list(batters['pitch_type'].unique())
    pitch_avgs = pd.DataFrame()
    for pitch in pitch_list:
        subset = rates.query('pitch_type == @pitch').reset_index().drop(columns='index')
        for q in range(4,(rates.shape[1]-2)):
                avg = 0
                stat = pd.DataFrame(pd.Series(subset.columns[q]),columns=['stat'])
                for p in range(0,len(subset)):
                    if q == 15:
                        x =  (subset.iloc[:,q][p])*(subset.iloc[:,-2][p]/sum(subset.iloc[:,-2]))
                        avg = avg+x
                    else:
                        x =  (subset.iloc[:,q][p])*(subset.iloc[:,-1][p]/sum(subset.iloc[:,-1]))
                        avg = avg+x
                avg = pd.DataFrame(pd.Series(round(avg,3)),columns=['value'])
                pitch_t = pd.DataFrame(pd.Series(pitch),columns=['pitch'])
                data = pd.concat([stat,avg,pitch_t],axis=1)
                pitch_avgs = pitch_avgs.append(data)

    pitch_avgs = pitch_avgs.pivot_table(index='pitch', columns='stat', values='value', aggfunc='first')
    pitch_avgs = pitch_avgs.reset_index()
    pitch_avgs = pitch_avgs.reset_index().drop(columns='index')
    for i in range(1,pitch_avgs.shape[1]-1):
        pitch_avgs[pitch_avgs.columns[i]] = round(pitch_avgs[pitch_avgs.columns[i]],1) 

    # get averages for the league environment

    just_stats = rates.iloc[:,4:rates.shape[1]-2]

    for i in range(0,len(just_stats)):
        pitch = rates.iloc[:,3][i]
        bucket_subset = pitch_avgs.query('pitch == @pitch')
        if i == 0:
            new_stats = round(just_stats.loc[i]/(bucket_subset.iloc[:,1:bucket_subset.shape[1]]),2)
        else:
            data = round(just_stats.loc[i]/(bucket_subset.iloc[:,1:bucket_subset.shape[1]]),2)
            new_stats = new_stats.append(data)
    new_stats = new_stats.reset_index().drop(columns = 'index')
    new_stats[['player_name','hitter','year','pitch','count','bbe']] = rates[['player_name','hitter','year','pitch_type','count','bbe']]
    new_stats = new_stats.rename(columns={'hitter':'playerid'})
    new_stats = new_stats.drop(columns='swing%')
    new_stats["player_name"] = [" ".join(n.split(", ")[::-1]) for n in new_stats["player_name"]]
    
    # using different dataset with much more data to run regression for stats
    # tests already run, this regression is well-fit
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LassoCV
    X = old_hits.iloc[:,:11].drop(columns='hr')
    y = old_hits.hr
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)
    result = LassoCV(cv=5, random_state=79, max_iter=10000)
    result = result.fit(X_train, y_train)
    
    """now we will get the predicted hr ratings for both a per-pitch basis
       and a players entire performance vs all pitches. Note that, since the 
       2025 data is more recent, we will be weighing those stats equally, even
       though there is a smaller sample size
       
       We will need the players stats vs all pitches becuase, for pitches they
       have barely faced, we will be regressing their stats to the mean, using
       both the leage averages on those pitches, which is 1, as well as their 
       own average stats"""
       
    # performace vs all pitches
    names = new_stats[['player_name','playerid']]
    names = names.drop_duplicates(subset=['player_name','playerid'], keep='first').reset_index(drop=True)
    weighted = pd.DataFrame()
    for i in range(0,len(names)):
        name = names['playerid'][i]
        player = new_stats.query('playerid == @name').reset_index().drop(columns='index')
        for q in range(0,len(player)):
            for p in range(0,11):
                year = player.year[q]
                player.iloc[:,p][q] = round(player.iloc[:,p][q]*(player['bbe'][q]/sum(player[player['year'] == year]['bbe'])),5)
        player = player.groupby(['player_name','year','playerid']).agg({
            **{col: 'sum' for col in list(player.columns[:11])},
            **{col: 'sum' for col in player.columns[15:17]}}).reset_index()
        if len(player['year'].unique()) == 2:
            player = player.groupby(['player_name','playerid']).agg({
                **{col: 'mean' for col in list(player.columns[3:14])},
                **{col: 'sum' for col in player.columns[14:]}}).reset_index()
        else:
            player.iloc[:,3:14] = (player.iloc[:,3:14]+1)/2
        player = round(player,2)
        weighted = weighted.append(player)
    weighted = weighted.reset_index(drop=True)
    whole_stats = weighted.drop(columns=['hr'])
    whole_stats['pred_hr'] = result.predict(whole_stats.iloc[:,2:12]).round(2)
    

    # now stats on a per-pitch basis

    per_pitch_short = pd.DataFrame()
    options = new_stats[['player_name','playerid','pitch','year','bbe']]
    options = options.drop_duplicates(subset=['player_name','playerid','pitch'], keep='first').reset_index(drop=True)
    new_stats = new_stats.drop(columns='hr')
    for i in range(0,len(options)):
        bbe = options['bbe'][i]
        name = options['playerid'][i]
        pitch_type = options['pitch'][i]
        pitch = new_stats.query('playerid == @name and pitch == @pitch_type').reset_index(drop=True)
        player = whole_stats.query('playerid == @name').reset_index(drop=True)
        if len(pitch) > 1:
        # Get BBE values for both years before grouping
            bbe_2024 = pitch.query('year == 2024')['bbe'].iloc[0] if len(pitch.query('year == 2024')) > 0 else 0
            bbe_2025 = pitch.query('year == 2025')['bbe'].iloc[0] if len(pitch.query('year == 2025')) > 0 else 0
            
            # Calculate weights based on BBE scenarios
            if bbe_2024 > 50 and bbe_2025 > 50:
                # Both over 50
                weight_2025 = 0.67 + 0.23 * (bbe_2025 / (bbe_2024 + bbe_2025))
            elif bbe_2024 > 50 and bbe_2025 <= 50:
                # 2024 > 50, 2025 <= 50
                weight_2025 = 0.25 + 0.65 * (bbe_2025 / (bbe_2024 + bbe_2025))
            elif bbe_2025 > 50 and bbe_2024 <= 50:
                # 2025 > 50, 2024 <= 50
                weight_2025 = 0.75 + 0.15 * (bbe_2025 / (bbe_2024 + bbe_2025))
            else:
                # Both under 50 - square root option with max 0.9
                sqrt_2024 = np.sqrt(bbe_2024) if bbe_2024 > 0 else 0
                sqrt_2025 = np.sqrt(bbe_2025) if bbe_2025 > 0 else 0
                if sqrt_2024 + sqrt_2025 > 0:
                    raw_weight_2025 = sqrt_2025 / (sqrt_2024 + sqrt_2025)
                    weight_2025 = min(raw_weight_2025, 0.9)
                else:
                    weight_2025 = 0.5
            
            weight_2024 = 1 - weight_2025
            
            # Apply weighted averages to the first 10 columns
            pitch_2024 = pitch.query('year == 2024').reset_index(drop=True)
            pitch_2025 = pitch.query('year == 2025').reset_index(drop=True)
            
            # Create weighted combination
            for col_idx in range(10):
                col_name = pitch.columns[col_idx]
                if col_name not in ['player_name', 'playerid', 'pitch', 'year']:
                    val_2024 = pitch_2024[col_name].iloc[0] if len(pitch_2024) > 0 else 0
                    val_2025 = pitch_2025[col_name].iloc[0] if len(pitch_2025) > 0 else 0
                    weighted_val = (val_2024 * weight_2024 + val_2025 * weight_2025).round(2)
                    pitch.loc[0, col_name] = weighted_val
            
            # Keep the first row and sum the counting stats
            pitch = pitch.iloc[:1].copy()
            for col in pitch.columns[14:]:
                pitch[col] = pitch_2024[col].sum() + pitch_2025[col].sum() if len(pitch_2024) > 0 and len(pitch_2025) > 0 else pitch[col]
            
            grouped = 1
        else:
            grouped = 0
        pitch = pitch.drop(columns='year')
        if bbe < 50:
            base_weight = (50 - bbe) / 50
            player_confidence = min(player.bbe[0] / 200, 1.0)
            player_weight = base_weight * (0.5 + 0.4 * player_confidence)
            league_weight = base_weight - player_weight
            for p in range(0,10):
                if grouped == 1:
                    pitch.iloc[:,p] = (pitch.iloc[:,p][0]*(1-base_weight) + player.iloc[:,p+2][0]*(player_weight) + 1*(league_weight)).round(2)
                else:
                    pitch.iloc[:,p] = (pitch.iloc[:,p][0]*(bbe/50) + player.iloc[:,p+2][0]*(player_weight) + 1*(league_weight)).round(2)
        per_pitch_short = per_pitch_short.append(pitch)

    per_pitch_short['pred_hr'] = result.predict(per_pitch_short.iloc[:,:10]).round(2)
    per_pitch_short = per_pitch_short.reset_index(drop=True)
    
    
    """Now that we have all players stats, we must get their splits based on
    the handedness of the pitcher they're facing so that can be factored in
    
    pretty much doing the exact same process but breaking up by handedness
    instead of pitch_type"""
    
    splits = batters.groupby(['player_name','hitter', 'p_throws','stand']).agg(
        hh=('hh', 'sum'), # Count of batted balls with exit velocity over 95 mph
        bbe=('bbe','sum'),
        barrels =('barrel','sum'),
        poorly_hit=('weak','sum'),
        fly_ball =('fly_ball','sum'),
        ground_ball =('ground_ball','sum'),
        line_drive =('line_drive','sum'),
        whiff =('whiff','sum'),
        swing =('swing','sum'),
        home_run =('home_run','sum'),
        pitch_count=('pitch_type', 'size')
        ).reset_index()
    xsplits = fbb.groupby(['player_name','hitter', 'p_throws','stand']).agg(
        tot_wob =('estimated_woba_using_speedangle','sum'),
        count=('player_name','size')).reset_index()
    xsplits['xwobacon'] = round(xsplits['tot_wob']/xsplits['count'],3)
    xsplits = xsplits[['player_name','hitter','p_throws','xwobacon','stand']]
    splits = splits.merge(xsplits,on=['player_name','hitter','p_throws','stand'])
    
    splits[['ev','la']] = batters.groupby(['player_name', 'hitter', 'stand', 'p_throws'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(splits.set_index(['player_name', 'hitter', 'stand', 'p_throws']).index).values
    
    # turn counting stats into rate stats

    rates = splits[['player_name','hitter','p_throws','stand','xwobacon','la','ev']]
    rates[['whiff','hh','fb','gb','ld','barrel','weak','hr','swing%','count','bbe']] = 0
    #split_metrics[[averages.columns[[range(2,len(averages.columns))]]]] = 0
    for i in range(0,len(splits)):
        rates['whiff'][i] = round(splits['whiff'][i]/splits['swing'][i],3)*100
        rates['hh'][i] = round(splits['hh'][i]/splits['bbe'][i],3)*100
        rates['gb'][i] = round((splits['ground_ball'][i])/splits['bbe'][i],3)*100
        rates['fb'][i] = round((splits['fly_ball'][i])/splits['bbe'][i],3)*100
        rates['ld'][i] = round((splits['line_drive'][i])/splits['bbe'][i],3)*100
        rates['hr'][i] = round((splits['home_run'][i])/splits['bbe'][i],3)*100
        rates['barrel'][i] = round((splits['barrels'][i])/splits['bbe'][i],3)*100
        rates['weak'][i] = round((splits['poorly_hit'][i])/splits['bbe'][i],3)*100
        rates['swing%'][i] = round((splits['swing'][i])/splits['pitch_count'][i],3)*100
        rates['count'][i] = splits['pitch_count'][i]
        rates['bbe'][i] = splits['bbe'][i]
    
    
    # getting averages for every pitch
    split_avgs = pd.DataFrame()
    matchups = splits[['stand','p_throws']].drop_duplicates()
    for i in range(0,4):
        stands = matchups['stand'][i]
        throws = matchups['p_throws'][i]
        subset = rates.query('stand == @stands and p_throws == @throws').reset_index().drop(columns='index')
        for q in range(4,(rates.shape[1]-2)):
                avg = 0
                stat = pd.DataFrame(pd.Series(subset.columns[q]),columns=['stat'])
                for p in range(0,len(subset)):
                    if q == 15:
                        x =  (subset.iloc[:,q][p])*(subset.iloc[:,-2][p]/sum(subset.iloc[:,-2]))
                        avg = avg+x
                    else:
                        x =  (subset.iloc[:,q][p])*(subset.iloc[:,-1][p]/sum(subset.iloc[:,-1]))
                        avg = avg+x
                avg = pd.DataFrame(pd.Series(round(avg,3)),columns=['value'])
                mtc = pd.DataFrame(matchups.loc[i]).T.reset_index(drop=True)
                data = pd.concat([stat,avg,mtc],axis=1)
                split_avgs = split_avgs.append(data)

    split_avgs = split_avgs.pivot_table(index=['stand','p_throws'], columns='stat', values='value', aggfunc='first')
    split_avgs = split_avgs.reset_index()
    split_avgs = split_avgs.reset_index().drop(columns='index')
    for i in range(2,split_avgs.shape[1]-1):
        split_avgs[split_avgs.columns[i]] = round(split_avgs[split_avgs.columns[i]],1)
    
    just_stats = rates.iloc[:,4:rates.shape[1]-2]

    for i in range(0,len(just_stats)):
        stands = rates.iloc[:,3][i]
        throws = rates.iloc[:,2][i]
        bucket_subset = split_avgs.query('stand == @stands and p_throws == @throws')
        if i == 0:
            new_stats = round(just_stats.loc[i]/(bucket_subset.iloc[:,2:bucket_subset.shape[1]]),2)
        else:
            data = round(just_stats.loc[i]/(bucket_subset.iloc[:,2:bucket_subset.shape[1]]),2)
            new_stats = new_stats.append(data)
    new_stats = new_stats.reset_index().drop(columns = 'index')
    new_stats[['player_name','hitter','stand','p_throws','count','bbe']] = rates[['player_name','hitter','stand','p_throws','count','bbe']]
    new_stats = new_stats.rename(columns={'hitter':'playerid'})
    new_stats = new_stats.drop(columns='swing%')
    new_stats["player_name"] = [" ".join(n.split(", ")[::-1]) for n in new_stats["player_name"]]
        
    
    names = new_stats[['player_name','playerid']]
    names = names.drop_duplicates(subset=['player_name','playerid'], keep='first').reset_index(drop=True)
    weighted = pd.DataFrame()
    for i in range(0,len(names)):
        name = names['playerid'][i]
        player = new_stats.query('playerid == @name').reset_index().drop(columns='index')
        for q in range(0,len(player)):
            for p in range(0,11):
                player.iloc[:,p][q] = round(player.iloc[:,p][q]*(player['bbe'][q]/sum(player['bbe'])),5)
        player = player.groupby(['player_name','stand','p_throws','playerid']).agg({
            **{col: 'sum' for col in list(player.columns[:11])},
            **{col: 'sum' for col in player.columns[15:17]}}).reset_index()
        if len(player) >= 2:
            player = player.groupby(['player_name','playerid']).agg({
                **{col: 'sum' for col in list(player.columns[4:])},}).reset_index()
        else:
            player.iloc[:,4:15] = (player.iloc[:,4:15]+1)/2
        player = round(player,2)
        weighted = weighted.append(player)
    weighted = weighted.reset_index(drop=True)
    whole_stats = weighted.drop(columns=['hr','stand','p_throws'])
    whole_stats['pred_hr'] = result.predict(whole_stats.iloc[:,2:12]).round(2)
    
    per_pitch_split = pd.DataFrame()
    options = new_stats[['player_name','playerid','stand','p_throws','bbe']]
    options = options.drop_duplicates(subset=['player_name','playerid','p_throws','stand'], keep='first').reset_index(drop=True)
    new_stats = new_stats.drop(columns='hr')
    for i in range(0,len(options)):
        bbe = options['bbe'][i]
        name = options['playerid'][i]
        throws = options['p_throws'][i]
        pitch = new_stats.query('playerid == @name and p_throws == @throws').reset_index(drop=True)
        player = whole_stats.query('playerid == @name').reset_index(drop=True)
        if bbe < 50:
            base_weight = (50 - bbe) / 50
            player_confidence = min(player.bbe[0] / 200, 1.0)
            player_weight = base_weight * (0.5 + 0.4 * player_confidence)
            league_weight = base_weight - player_weight
            for p in range(0,10):
                pitch.iloc[:,p] = (pitch.iloc[:,p][0]*(bbe/33) + player.iloc[:,p+2][0]*(player_weight) + 1*(league_weight)).round(2)
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
        plat_diff = dis.iloc[:,:10]-adv.iloc[:,:10]
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
