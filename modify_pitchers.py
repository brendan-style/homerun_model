# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:47:42 2025

@author: Brendan
"""

def modify_pitchers(pitchers, old_pitch,players,ids):
    import pandas as pd
    from numpy import select
    from unidecode import unidecode
    import numpy as np
    fbb = pitchers.query('bbe==1')
    fbb = fbb.dropna(subset=['estimated_woba_using_speedangle'])
    
    # getting average bf and outs gotten for every starter
    
    outings = pitchers.groupby(['player_name','pitcher','batter','game_date','inning','outs_when_up','events']).agg(
        pitches=('batter','size'))
    outings = outings.reset_index()
    outings['outs'] = ((outings['inning']*3)-3) + outings['outs_when_up']
    outings = outings.groupby(['player_name','pitcher','game_date']).agg(
        bf=('game_date', 'size'),
        last_out=('outs','max'),
        first_out=('outs','min'))
    outings = outings.reset_index()
    outings['outs'] = outings.last_out - outings.first_out
    outings = outings.drop(columns=['first_out','last_out'])
    pitch_counts = pitchers.groupby(['pitcher', 'game_date']).size().reset_index(name='pitch_count')
    outings = outings.merge(pitch_counts, on=['pitcher', 'game_date'], how='left')
    
    last_events = pitchers.groupby(['pitcher', 'game_date'])['events'].last().reset_index(name='last_event')
    outings = outings.merge(last_events, on=['pitcher', 'game_date'], how='left')
    
    outings['outs'] += outings['last_event'].str.contains('triple_play', na=False).astype(int) * 3
    outings['outs'] += outings['last_event'].str.contains('double_play', na=False).astype(int) * 2
    outings['outs'] += (outings['last_event'].str.contains('out', na=False) | outings['last_event'].str.contains('sac', na=False)).astype(int) * 1
    outings = outings.drop(columns=['last_event'])
    
    outings = outings.groupby(['player_name','pitcher']).agg(
        avg_bf=('bf','mean'),
        std_bf=('bf','std'),
        avg_outs=('outs','mean'),
        atd_outs=('outs','std'),
        avg_pc=('pitch_count','mean'),
        std_pc=('pitch_count','std'),
        apps=('pitcher','size')).reset_index()
    outings = outings.round(1)
    outings["player_name"] = [" ".join(n.split(", ")[::-1]) for n in outings["player_name"]]
    outings = outings.rename(columns={'pitcher':'playerid'})
    
    
    grouped = pitchers.groupby(['player_name','pitcher', 'year', 'pitch_type']).agg(
        velo=('release_speed', 'mean'),
        spin_rate=('release_spin_rate', 'mean'),
        hh=('hh', 'sum'), # Count of batted balls with exit velocity over 95 mph
        bbe=('bbe','sum'),
        barrels =('barrel','sum'),
        poorly_hit=('weak','sum'),
        fly_ball =('fly_ball','sum'),
        ground_ball =('ground_ball','sum'),
        line_drive =('line_drive','sum'),
        whiff =('whiff','sum'),
        rls_avg =('release_pos_x','mean'),
        rls_std =('release_pos_x','std'),
        swing =('swing','sum'),
        x_move =('pfx_x','mean'),
        z_move =('pfx_z','mean'),
        extension =('release_extension','mean'),
        home_run =('home_run','sum'),
        pitch_count=('pitch_type', 'size')
        ).reset_index()
    xgrouped = fbb.groupby(['player_name','pitcher', 'year', 'pitch_type']).agg(
        tot_wob =('estimated_woba_using_speedangle','sum'),
        count=('pitch_type','size')).reset_index()
    xgrouped['xwobacon'] = round(xgrouped['tot_wob']/xgrouped['count'],3)
    xgrouped = xgrouped[['player_name','pitcher','year','pitch_type','xwobacon']]
    grouped = grouped.merge(xgrouped,on=['player_name','pitcher','year','pitch_type'])
    #grouped = grouped.dropna(subset=['velo','spin_rate'])
    grouped = grouped.round({'spin_rate': 0, 'velo': 1, 'rls_avg': 2, 'rls_std': 2})

    grouped[['ev', 'la']] = pitchers.groupby(['player_name', 'pitcher', 'year', 'pitch_type'])[['launch_speed', 'launch_angle']].mean().round(1).reindex(grouped.set_index(['player_name', 'pitcher', 'year', 'pitch_type']).index).values
    
    # turn counting stats into rate stats

    rates = grouped[['player_name','pitcher','year','pitch_type','velo','spin_rate','xwobacon','la','ev','rls_avg','rls_std','x_move','z_move','extension']]
    rates[['whiff','hh','fb','gb','ld','barrel','weak','hr','swing%','count','bbe']] = 0
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
        
    rates = rates.round({'x_move': 2, 'z_move': 2, 'extension': 1})
    # getting averages for every pitch
    pitch_list = list(pitchers['pitch_type'].unique())
    
    # getting absolute value of horizontal movement since otherwise it would cancel out
    
    rates['x_move'] = abs(rates['x_move'])
    rates = rates.dropna().reset_index(drop=True)
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
    pitch_avgs = pitch_avgs.drop(columns=['rls_std','rls_avg'])
    rates = rates.drop(columns=['rls_std','rls_avg'])
    pitch_avgs = pitch_avgs.dropna()
    for i in range(1,pitch_avgs.shape[1]):
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
    new_stats[['player_name','pitcher','year','pitch','count','bbe']] = rates[['player_name','pitcher','year','pitch_type','count','bbe']]
    new_stats = new_stats.rename(columns={'pitcher':'playerid'})
    new_stats = new_stats.drop(columns='swing%')
    new_stats["player_name"] = [" ".join(n.split(", ")[::-1]) for n in new_stats["player_name"]]
    
    # using different dataset with much more data to run regression for stats
    # tests already run, this regression is well-fit
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LassoCV
    X = old_pitch.iloc[:,:16].drop(columns='hr')
    y = old_pitch.hr
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
            for p in range(0,16):
                year = player.year[q]
                player.iloc[:,p][q] = round(player.iloc[:,p][q]*(player['bbe'][q]/sum(player[player['year'] == year]['bbe'])),5)
        player = player.groupby(['player_name','year','playerid']).agg({
            **{col: 'sum' for col in list(player.columns[:16])},
            **{col: 'sum' for col in player.columns[20:]}}).reset_index()
        if len(player['year'].unique()) == 2:
            player = player.groupby(['player_name','playerid']).agg({
                **{col: 'mean' for col in list(player.columns[3:19])},
                **{col: 'sum' for col in player.columns[19:]}}).reset_index()
        else:
            player.iloc[:,3:14] = (player.iloc[:,3:14]+1)/2
        player = round(player,2)
        weighted = weighted.append(player)
    weighted = weighted.reset_index(drop=True)
    whole_stats = weighted.drop(columns=['hr','year'])
    whole_stats['pred_hr'] = result.predict(whole_stats.iloc[:,2:17]).round(2)
    

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
            grouped= 1
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
                 
                # Create weighted combination
            for col_idx in range(10):
                col_name = pitch.columns[col_idx]
                if col_name not in ['player_name', 'playerid', 'pitch', 'year']:
                    val_2024 = pitch[col_name].iloc[0]
                    val_2025 = pitch[col_name].iloc[1]
                    weighted_val = (val_2024 * weight_2024 + val_2025 * weight_2025).round(2)
                    pitch.loc[0, col_name] = weighted_val
            for col in pitch.columns[19:]:
                pitch[col] = sum(pitch[col])
            pitch = pitch.iloc[:1].copy()
        else:
            grouped = 0
        pitch = pitch.drop(columns='year')
        if bbe < 50:
            base_weight = (50 - bbe) / 50
            player_confidence = min(player.bbe[0] / 200, 1.0)
            player_weight = base_weight * (0.5 + 0.4 * player_confidence)
            league_weight = base_weight - player_weight
            for p in range(0,15):
                if grouped ==1:
                    pitch.iloc[:,p] = (pitch.iloc[:,p][0]*(1-base_weight) + player.iloc[:,p+2][0]*(player_weight) + 1*(league_weight)).round(2)
                else:
                    pitch.iloc[:,p] = (pitch.iloc[:,p][0]*(bbe/33) + player.iloc[:,p+2][0]*(player_weight) + 1*(league_weight)).round(2)
        per_pitch_short = per_pitch_short.append(pitch)

    per_pitch_short['pred_hr'] = result.predict(per_pitch_short.iloc[:,:15]).round(2)
    
    players['team_id'] = players['team_id'].astype(int)
    outings = outings.rename(columns={'pitcher':'playerid'})
    per_pitch_short = per_pitch_short.merge(outings,how='inner',on=['playerid','player_name'])
    players = players.merge(ids, on='team_id', how='left')
    players = players[['person_id','Stadium','person_full_name']]
    players = players.rename(columns={'person_id':'playerid','person_full_name':'player_name'})
    players['playerid'] = players['playerid'].astype(int)
    per_pitch_short = per_pitch_short.merge(players,how='inner',on=['playerid','player_name'])
    per_pitch_short = per_pitch_short.drop_duplicates()
    
    per_pitch_short['player_name'] = per_pitch_short['player_name'].apply(unidecode)
    per_pitch_short = per_pitch_short.reset_index(drop=True)
    
    """since we cannot aggregate things like velo, movement, and spin rate,
    as a pitcher's goal is to vary those things, we will not be getting splits
    for pitchers"""
    
    
    
    return per_pitch_short
