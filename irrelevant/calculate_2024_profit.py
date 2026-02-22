# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 20:15:23 2026

@author: Brendan
"""

"""
in this file, we will be taking our 2024 home run odds that we pulled from
a betting website, calculating stats week by week (otherwise it would take forever),
and making selections before knowing what happened. In order to calculate stats, I will
have to use the bbe weights we calculated in a different file.

HAVE TO FIGURE OUT HOW TO DETERMINE STARTING PITCHER
"""
#%% edit df's (one time use)
import pandas as pd
odds = pd.read_csv('historical_odds_multi.csv')
odds = odds.groupby(['name','date']).filter(lambda x: len(x) > 1)
overs = odds.query('prop == "Over"')
unders = odds.query('prop == "Under"')
concise_df = overs.merge(unders,on=['name','away','home','date'],how='inner')
concise_df = concise_df.rename(columns={'odds_x':'over','odds_y':'under'})
concise_df = concise_df.drop(columns=['prop_x','prop_y'])
concise_df = concise_df[['name','date','home','away','over','under']]
concise_df[['first','last']] = concise_df['name'].str.split(' ',n=1, expand=True)
concise_df = concise_df.drop(columns='name')
concise_df.over = ((100/(concise_df.over+100))).round(3)
concise_df.under = ((concise_df.under/(concise_df.under-100))).round(3)
# noticed that some players had one off misrepresented odds, all were above 45%
#concise_df = concise_df.query('over < .45')
concise_df.to_csv('historical_odds_multi.csv',index=False)

# since no playerids, had to merge based on names

from unidecode import unidecode
matchups = pd.read_csv('historical_odds_multi.csv')
bat_names = pd.read_csv('batters.csv').query('game_year >= 2021 and game_year < 2025')[['player_name','playerid']].drop_duplicates().reset_index(drop=True)
bat_names[['last','first']] = bat_names['player_name'].str.split(', ',expand = True)
bat_names['player_name'] = bat_names['first']+' '+bat_names['last']
bat_names = bat_names.drop(columns=['first','last']).rename(columns={'playerid':'batterid'})
bat_names['player_name'] = bat_names['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
matchups['player_name'] = matchups['first']+' '+matchups['last']
matchups['player_name'] = matchups['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
matchups = matchups.drop(columns=['first','last'])
matchups = matchups.merge(bat_names,how='left',on='player_name').dropna()
matchups.date = pd.to_datetime(matchups.date)
matchups.date = matchups['date'].dt.strftime('%Y-%m-%d')
matchups.to_csv('busted_sgo_odds.csv',index=False)

# getting starting pitchers, HR results, and team affiliations

matchups[['team','pitcher','hr','pa','stand','p_throws','home','away']] = 0
#%%
from pandas.errors import ParserError
import time
import pybaseball as bb


for i in range(len(matchups)):
    date = matchups.date[i]
    try:
        stats = bb.statcast_batter(date,date,matchups.batterid[i]).sort_values(by=['inning','outs_when_up','balls','strikes'],ascending=True)[['home_team','away_team','pitcher','events','inning_topbot','inning','outs_when_up','stand','p_throws']].reset_index(drop=True)
    except ParserError:
        time.sleep(1)
        stats = bb.statcast_batter(date,date,matchups.batterid[i]).sort_values(by=['inning','outs_when_up','balls','strikes'],ascending=True)[['home_team','away_team','pitcher','events','inning_topbot','inning','outs_when_up','stand','p_throws']].reset_index(drop=True)
    if stats.empty:
        matchups = matchups.drop(i)
        continue
    first_inning = stats.inning[0]
    if first_inning > 3:
        matchups = matchups.drop(i)
        continue
    matchups.pa[i] = len(stats[['inning','outs_when_up']].drop_duplicates())
    matchups.home[i] = stats.home_team[0]
    matchups.away[i] = stats.away_team[0]
    if stats.inning_topbot[0] == 'Bot':
        matchups.team[i] = matchups.home[i]
    else:
        matchups.team[i] = matchups.away[i]

    if 'home_run' in list(stats.events.unique()):
        matchups.hr[i] = 1
    else:
        pass
    matchups.pitcher[i] = stats.pitcher[0]
    matchups.stand[i] = stats.stand[0]
    matchups.p_throws[i] = stats.p_throws[0]
matchups = matchups.reset_index(drop=True)
#%% 
import pandas as pd
matchups = pd.read_csv('busted_sgo_odds.csv')
# find double headers - this may eliminate some extra innings games but that's okay
double_h = matchups.groupby(['date','away','home']).agg(max_pa=('pa','max'))
games_to_drop = double_h.query('max_pa > 6')
test = double_h[double_h.index.isin(games_to_drop.index)].reset_index()
test_too = matchups.merge(test,how='outer',on=['date','home','away'])
test_too = test_too.fillna(0)
test_too = test_too.query('max_pa == 0').drop(columns='max_pa')
test_too.to_csv('busted_sgo_odds.csv',index=False)
#%% give them lineup spots
import pandas as pd
from unidecode import unidecode
factors = pd.read_excel('HR_factors.xlsx')
matchups_23 = pd.read_csv('historical_odds_multi.csv')
matchups_24 = pd.read_csv('historical_odds_24.csv')
matchups = matchups_23.append(matchups_24)
lineups = pd.read_csv('all_boxscores.csv')
matchups.player_name = matchups.player_name.apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
lineups.player = lineups.player.apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
factors = factors[['team','city']].drop_duplicates()

matchups = matchups.merge(factors,on='team')
matchups['full_team'] = matchups.city + " " + matchups.team
matchups = matchups.sort_values(by=['date','full_team']).reset_index(drop=True)
lineups = lineups.rename(columns={'team':'full_team','player':'player_name'})
matchups = matchups.merge(lineups,on=['player_name','full_team','date'],how='left')
matchups = matchups.dropna(subset='#').drop_duplicates()
matchups = matchups.drop(columns=['city','full_team'])
archives = pd.read_excel('archive.xlsx')
archives = archives[['date','home_team','away_team','Over','Under','player','playerid','team','pitcher','hr','handedness','p_throws','lineup_spot']]
archives = archives.rename(columns={'home_team':'home','away_team':'away','Over':'over','Under':'under','player':'player_name','handedness':'stand','lineup_spot':'#','playerid':'batterid'})
archives[['first','last']] = archives['pitcher'].str.split(' ',n=1, expand=True)
names = archives[['first','last']].drop_duplicates().reset_index(drop=True)
names['pitcher_id'] = 0
for i in range(len(names)):
    x = bb.playerid_lookup(names['last'][i],names['first'][i],fuzzy=True)
    for q in range(len(x)):
        if x.mlb_played_last[q] == 2025:
            names['pitcher_id'][i] = x.key_mlbam[q]
        else:
            continue
archives = archives.merge(names,on=['first','last'])
archives = archives.drop(columns=['first','last','pitcher'])
archives = archives.rename(columns={'pitcher_id':'pitcher'})
matchups = matchups.drop(columns='pa')
factors = factors[['Stadium','team']].drop_duplicates()
archives = archives.rename(columns={'team_x':'Stadium'})
archives = archives.merge(factors,on='Stadium')
archives = archives.drop(columns='Stadium')
full = matchups.append(archives)
# luis_garcia = 677651
full.to_csv('all_matchups.csv',index=False)
#%%
import pandas as pd
import pybaseball as bb
from datetime import timedelta,datetime
from modify_pitchers import modify_pitchers
from modify_batters import modify_batters
from unidecode import unidecode
matchups = pd.read_csv('2025_filtered_lineup_odds.csv')
factors = pd.read_excel('HR_factors.xlsx') 
# Get batters
date = '2025-04-01'
batters = pd.read_csv('batters.csv')
old_hits = pd.read_csv('old_hits.csv')
hit_stats = modify_batters(batters,old_hits,date)
# pitchers
old_pitch = pd.read_csv('old_pitch.csv')
pitchers = pd.read_csv('pitchers.csv')

pitch_stats = modify_pitchers(pitchers,old_pitch,date)

hit_stats.pred_hr = hit_stats.pred_hr.clip(lower=0.01)
pitch_stats.pred_hr = pitch_stats.pred_hr.clip(lower=0.01)
# Match playerid's


# get rosters and match teams / playerid to matchups sheet
#%%
import subprocess
import tempfile
import os
import pandas as pd
from io import StringIO
from unidecode import unidecode
from pandas.errors import ParserError
import time


"""
date = matchups.date[0]
date = datetime.strptime(date,'%m/%d/%Y') + timedelta(days=7)
date = datetime.strftime(date,'%m/%d/%Y')
weekly = matchups.query('date <= @date').reset_index(drop=True)
"""
from statistics import mean
from scipy import stats
from math import floor,ceil
pa_mod = {0:0,1:0.44,2:1.21,3:1.36,4:1.50,5:1.5}
pa_per_game = {1: 4.65,2: 4.55,3: 4.43,4: 4.33,5: 4.24,6: 4.13,7: 4.01,8: 3.90,9: 3.77}
final_ratings=pd.DataFrame()
week_pacer = 0
dates_list = list(matchups['date'].unique())
dates_list = dates_list[93:]

for date in dates_list:
    c_year = int(date[:4])
    daily = matchups.query('date == @date').reset_index(drop=True)
    if daily.empty:
        continue
    else:
        pass
    if date >= '2025-04-01' and date <= '2025-04-03':
        pass
    elif date == '2025-04-26':
        hit_stats = modify_batters(batters,old_hits,date)
        pitch_stats = modify_pitchers(pitchers,old_pitch,date)
        hit_stats.pred_hr = hit_stats.pred_hr.clip(lower=0.01)
        pitch_stats.pred_hr = pitch_stats.pred_hr.clip(lower=0.01)         
    else:
        if week_pacer%7 == 0:
            hit_stats = modify_batters(batters,old_hits,date)
            pitch_stats = modify_pitchers(pitchers,old_pitch,date)
            hit_stats.pred_hr = hit_stats.pred_hr.clip(lower=0.01)
            pitch_stats.pred_hr = pitch_stats.pred_hr.clip(lower=0.01)          
        else:
            pass
    week_pacer += 1
        
    r_code = f"""
    library(baseballr)
    team_ids <- c(108:121, 133:147, 158)
    all_rosters <- lapply(team_ids, function(x) {{ roster <- try(mlb_rosters(team_id = x, season = 2024, roster_type = 'active', date='{date}'), silent = TRUE); if(!inherits(roster, "try-error")) {{ roster$team_id <- x; roster }} }})
    combined_rosters <- do.call(rbind, all_rosters[!sapply(all_rosters, is.null)])
    write.csv(combined_rosters, stdout(), row.names = FALSE)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        temp_script = f.name
    result = subprocess.run([r'C:\Program Files\R\R-4.2.1\bin\Rscript.exe', temp_script], 
                           capture_output=True, text=True, check=True, encoding='utf-8')
    os.unlink(temp_script)
    players = pd.read_csv(StringIO(result.stdout))
    players = players.applymap(lambda x: unidecode(str(x)) if pd.notna(x) else x)
    players = players.rename(columns={'person_id':'playerid'})
    players[['team_id','playerid']] = players[['team_id','playerid']].astype(int)
    players = players.merge(factors,how='left',on='team_id')
    players = players[['team_id','playerid','team']].drop_duplicates()
    
    pitch_daily = pitch_stats.merge(players,how='inner',on='playerid')
    hits_daily = hit_stats.merge(players,how='inner',on='playerid')
    teams = list(daily['team'].drop_duplicates())
    
    bullpen_stats = pd.DataFrame()
    for team in teams:
        scope = pitch_daily.query('team == @team')
        bf_values = sorted(scope.avg_bf.unique(), reverse=True)
        starter_line = bf_values[4]
        scope = scope.query('avg_bf < @starter_line') # drops highest 5 which is the rotation in many cases
        if scope.empty:
            scope = pitch_daily.query('team == @team')
            scope = scope.query('avg_bf < 9')
        pitch_list = list(scope.pitch_type.unique())
        team_bps = pd.DataFrame()
        for pitch in pitch_list:
            pbp = scope.query('pitch_type == @pitch')
            pbp = pbp.groupby('pitch_type').agg(
                pred_hr = ('pred_hr','mean'),
                pitch_count = ('pitch_count','sum')).reset_index().round(2)
            pbp['team'] = team
            team_bps = team_bps.append(pbp)
        team_bps = team_bps.reset_index(drop=True)
        total_count = team_bps['pitch_count'].sum()
        team_bps['percentage'] = (team_bps['pitch_count']/total_count).astype(float)
        bullpen_stats = bullpen_stats.append(team_bps)
    bullpen_stats = bullpen_stats.round(3)
    daily['rating'] = 0
    for i in range(len(daily)):
        b_name = daily.batterid[i]
        p_name = daily.pitcher[i]
        stadium = daily.home[i]
        pa_count=pa_per_game[daily['#'][i]]
        if daily.team[i] == daily.home[i]:
            opp_bp = daily.away[i]
        else:
            opp_bp = daily.home[i]
        bp = bullpen_stats.query('team == @opp_bp').reset_index(drop=True)
        bat = hit_stats.query('playerid == @b_name').reset_index(drop=True)
        pitch = pitch_stats.query('playerid == @p_name').reset_index(drop=True)
        if not bat.empty and not pitch.empty:
            matchup = bat.merge(pitch,how='outer',on='pitch_type').fillna(0)
            matchup['rating'] = 0
            matchup = matchup.sort_values(by='pred_hr_x',ascending = False)
            matchup = matchup.drop_duplicates(subset='pitch_type',keep='first').reset_index(drop=True)
            matchup['percentage'] = round(matchup.pitch_count/sum(matchup.pitch_count),3)
            for q in range(0,len(matchup)):
                bat_r = matchup['pred_hr_x'][q]
                pit_r = matchup['pred_hr_y'][q]
                amt = matchup['percentage'][q]
                if bat_r == 0:
                    real_pitches = matchup.query('pred_hr_x != 0')
                    matchup['rating'][q] = round((pit_r+(mean(real_pitches.pred_hr_x)*0.9))*amt,2)
                elif pit_r == 0:
                    continue
                else:
                    matchup['rating'][q] = round((pit_r++bat_r)*amt,2)
            rating = round(sum(matchup.rating),2)
        else: continue
        bf_mean = pitch.avg_bf[0]
        bf_std = pitch.std_bf[0]
        if bf_std != bf_std or bf_std == 0:
            bf_std = round(mean(pitch_stats.query('avg_bf >= 10.0')['std_bf'].dropna()),2)
        else:
            pass
        starter_pa = 0
        pa = 1
        starter_mod = 0
        spot = int(daily['#'][i])
        for p in range(spot,spot+37,9):
            zs1 = round((p-bf_mean)/bf_std,2)
            zs2 = round(((p+9)-bf_mean)/bf_std,2)        
            prob = round(stats.norm.cdf(zs2) - stats.norm.cdf(zs1),5)
            starter_mod += (pa_mod[pa]*prob)
            starter_pa += (pa*prob)
            pa += 1
        starter_pa = starter_pa.round(2)
        bp_pa = round(pa_count - starter_pa,2)
        bats = daily.stand[i]
        throws = daily.p_throws[i]
        if bats == "S" and throws == 'R':
            bats = 'L'
        elif bats == "S" and throws == 'L':
            bats = 'R'
        park = factors.query('Handedness == @bats and team == @stadium').reset_index(drop=True)
        rating = round(rating*park.HR[0],2)
        if bats == throws:
            rating = rating + matchup['plat_disc'][0]/2
        else:
            rating = rating + (matchup['plat_disc'][0]/2*(-1))
    
        rating = round(rating*starter_mod,2)
        matchup = bat.merge(bp,how='outer',on='pitch_type').fillna(0)
        matchup['rating'] = 0
        for q in range(0,len(matchup)):
            bat_r = matchup['pred_hr_x'][q]
            pit_r = matchup['pred_hr_y'][q]
            amt = matchup['percentage'][q]
            if bat_r == 0:
                real_pitches = matchup.query('pred_hr_x != 0')
                matchup['rating'][q] = round((pit_r+(mean(real_pitches.pred_hr_x)*0.9))*amt,2)
            elif pit_r == 0:
                continue
            else:
                matchup['rating'][q] = round((pit_r++bat_r)*amt,2)
        bp_rating = round(sum(matchup.rating),2)
    
        
        # cannot say whether or not there will be a platoon advantage (just yet)(could take the collective handedness)
        
        bp_rating = round(bp_rating*park.HR[0],2)
    
        bp_mod = ((ceil(bp_pa)-bp_pa).round(2)*pa_mod[floor(bp_pa)]) + ((bp_pa-floor(bp_pa)).round(2)*pa_mod[ceil(bp_pa)])
        bp_rating = round(bp_rating*bp_mod,2)
    
        hr_rating = (rating+bp_rating).round(2)
        daily['rating'][i] = hr_rating
    daily = daily.query('rating != 0')
    daily.rating = daily.rating.clip(lower= 0.1)
    final_ratings = final_ratings.append(daily)
    
final_ratings = final_ratings.dropna().drop_duplicates().reset_index(drop=True)
final_ratings.to_csv('matchup_25.csv',index=False)
# find double headers - this may eliminate some extra innings games but that's okay
double_h = final_ratings.groupby(['date','away','home']).agg(max_pa=('pa','max'))
games_to_drop = double_h.query('max_pa > 6')
test = double_h[double_h.index.isin(games_to_drop.index)].reset_index()
test_too = final_ratings.merge(test,how='outer',on=['date','home','away'])
test_too = test_too.fillna(0)
test_too = test_too.query('max_pa == 0').drop(columns=['max_pa','pa'])

final_ratings = final_ratings.drop(columns='pa')
#final_ratings.to_csv('historical_profit_calc_all.csv',index=False)

# If i have time, go back and remove pitches that they threw the previous season and not current
# something like if len(2024 pitches) > 200, and max(pitch_type's_year) != 2024, skip

   
#%%
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from testing_potency import backtest_unders, backtest_overs
from datetime import datetime
from statistics import mean, median, stdev
from numpy import sqrt
matchup_25 = pd.read_csv('matchup_25.csv')
final_ratings = pd.read_csv('historical_profit_calc_all.csv')
final_ratings = final_ratings.append(matchup_25).drop(columns='pa')
final_ratings = final_ratings.drop_duplicates(subset=['batterid','date','over'])
final_ratings = final_ratings.drop_duplicates(subset=['batterid','date','under'])
final_ratings = final_ratings.sort_values(by='rating',ascending = False).reset_index(drop=True)

final_ratings.rating = final_ratings.rating.astype(float)
final_ratings.rating = round(final_ratings.rating/2,2)
#high = round(mean(final_ratings.rating)+stdev(final_ratings.rating)*3,2)
#low = round(mean(final_ratings.rating)-stdev(final_ratings.rating)*3,2)
#high = final_ratings.rating[49]
#low = final_ratings.rating[len(final_ratings)-50]
import matplotlib.pyplot as plt
plt.hist(final_ratings.rating, bins=50, edgecolor='black')
plt.show()
#std_test = final_ratings.query('rating >= @high or rating <= @low')
#final_ratings = final_ratings.drop(std_test.index).reset_index(drop=True)

del matchup_25,#std_test,high, low


# ideas
# sqrt
# add in monthly effects
#%% testing different thresholds (OVERS)
# run again but keep best profit and best roi next time
best_over = pd.DataFrame()


for i in range(10,100,5):
    start = datetime.now()
    rating_weight = i/100
    final_ratings['over_rating'] = round((final_ratings.rating*rating_weight) + (final_ratings.over*(1-rating_weight)),3)
    #final_ratings['under_rating'] = round((final_ratings.rating*rating_weight) + (final_ratings.under*(1-rating_weight)),3)
    final_ratings['year'] = pd.to_datetime(final_ratings['date']).dt.year
    X = final_ratings['over_rating']
    y = final_ratings.hr
    # .reshape(-1,1)
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,train_size=.25,random_state=33)
    result = LogisticRegression().fit(X_tr.values.reshape(-1,1),y_tr)
    result.coef_
    pred = final_ratings[['over_rating']]
    final_ratings[['pred_under','pred_over']] = (result.predict_proba(pred.values.reshape(-1,1))).round(3)
    
    final_ratings['under_diff'] = final_ratings.pred_under-final_ratings.under
    final_ratings['over_diff'] =final_ratings.pred_over-final_ratings.over
    
    # resplit for validation
    
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,train_size=.40,random_state=33)
    
    final_ratings_train = final_ratings.loc[X_tr.index].reset_index(drop=True)
    final_ratings_test = final_ratings.loc[X_te.index].reset_index(drop=True)

    over_profit = backtest_overs(final_ratings_train)
    over_profit = over_profit[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']].rename(columns={'roi':'train_roi','total_profit':'train_profit','num_picks':'train_picks','win_rate':'train_win_rate'})
    over_test = backtest_overs(final_ratings_test)
    over_test = over_test[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']]
    both_over = over_profit.merge(over_test,on=['diff_threshold','pred_hr_threshold'])
    both_over['sum_profit'] = both_over.total_profit + both_over.train_profit
    both_over['sum_picks'] = both_over.num_picks + both_over.train_picks
    both_over['weight'] = i
    best_over = best_over.append(both_over)
    best_over = best_over.sort_values(by='sum_profit',ascending=False)
    time = datetime.now()-start 
    print(f'TIME FOR LOOP: {time}')
best_over = best_over.query('sum_picks >= 500')
best_over['both_roi'] = (best_over.sum_profit/(best_over.sum_picks*10))*100

best_over.to_csv('best_overs.csv',index=False)
#%% testing different thresholds (UNDERS)
best_under = pd.DataFrame()

for i in range(10,95,5):
    start = datetime.now()
    rating_weight = i/100
    #final_ratings['over_rating'] = round((final_ratings.rating*rating_weight) + (final_ratings.over*(1-rating_weight)),3)
    final_ratings['under_rating'] = round((final_ratings.rating*rating_weight) + (final_ratings.under*(1-rating_weight)),3)
    final_ratings['year'] = pd.to_datetime(final_ratings['date']).dt.year
    #final_ratings['month'] = pd.to_datetime(final_ratings['date']).dt.month
    X = final_ratings['under_rating']
    y = final_ratings.hr
    # .reshape(-1,1)
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,train_size=.25,random_state=33)
    result = LogisticRegression().fit(X_tr.values.reshape(-1,1),y_tr)
    result.coef_
    pred = final_ratings['under_rating']
    final_ratings[['pred_under','pred_over']] = (result.predict_proba(pred.values.reshape(-1,1))).round(3)
    
    final_ratings['under_diff'] = final_ratings.pred_under-final_ratings.under
    final_ratings['over_diff'] =final_ratings.pred_over-final_ratings.over
    
    # resplit for validation
    
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,train_size=.40,random_state=33)
    
    final_ratings_train = final_ratings.loc[X_tr.index].reset_index(drop=True)
    final_ratings_test = final_ratings.loc[X_te.index].reset_index(drop=True)

    under_profit = backtest_unders(final_ratings_train)
    under_profit = under_profit[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']].rename(columns={'roi':'train_roi','total_profit':'train_profit','num_picks':'train_picks','win_rate':'train_win_rate'})
    under_test = backtest_unders(final_ratings_test)
    under_test = under_test[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']]
    both_under = under_profit.merge(under_test,on=['diff_threshold','pred_hr_threshold'])
    both_under['sum_profit'] = both_under.total_profit + both_under.train_profit
    both_under['sum_picks'] = both_under.num_picks + both_under.train_picks
    both_under['weight'] = i
    best_under = best_under.append(both_under)
    best_under = best_under.sort_values(by='sum_profit',ascending=False)
    time = datetime.now()-start 
    print(f'TIME FOR LOOP: {time}')

best_under = best_under.query('sum_picks >= 700')
best_under['both_roi'] = (best_under.sum_profit/(best_under.sum_picks*10))*100


best_under.to_csv('best_unders.csv',index=False)

#%%
final_ratings['over_rating'] = round((final_ratings.rating*.5) + (final_ratings.over*.50),3)
final_ratings['under_rating'] = round((final_ratings.rating*.15) + (final_ratings.under*.85),3)
final_ratings['month'] = pd.to_datetime(final_ratings['date']).dt.month
final_ratings['year'] = pd.to_datetime(final_ratings['date']).dt.year
X = final_ratings[['over_rating','under_rating']]
y = final_ratings.hr
# .reshape(-1,1)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,train_size=.25,random_state=33)
result = LogisticRegression().fit(X_tr.values,y_tr.values)
result.coef_
pred = final_ratings[['over_rating','under_rating']]
final_ratings[['pred_under','pred_over']] = (result.predict_proba(pred.values)).round(3)
#final_ratings = final_ratings.drop(columns='pred_under')

# .reshape(-1,1)
"""
result = LogisticRegression().fit(final_ratings['under_rating'].values.reshape(-1,1),y)
result.coef_
pred = final_ratings.under_rating
final_ratings[['pred_under','pred_over_2']] = (result.predict_proba(pred.values.reshape(-1,1))).round(3)
final_ratings = final_ratings.drop(columns='pred_over_2')
"""

final_ratings['under_diff'] = final_ratings.pred_under-final_ratings.under
final_ratings['over_diff'] =final_ratings.pred_over-final_ratings.over

# resplit for validation

X_tr,X_te,y_tr,y_te = train_test_split(X,y,train_size=.40,random_state=33)

final_ratings_train = final_ratings.loc[X_tr.index].reset_index(drop=True)
final_ratings_test = final_ratings.loc[X_te.index].reset_index(drop=True)

under_profit = backtest_unders(final_ratings_train)
over_profit = backtest_overs(final_ratings_train)

under_profit = under_profit[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']].rename(columns={'roi':'train_roi','total_profit':'train_profit','num_picks':'train_picks','win_rate':'train_win_rate'})
over_profit = over_profit[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']].rename(columns={'roi':'train_roi','total_profit':'train_profit','num_picks':'train_picks','win_rate':'train_win_rate'})

under_test = backtest_unders(final_ratings_test)
over_test = backtest_overs(final_ratings_test)

under_test = under_test[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']]
over_test = over_test[['diff_threshold','pred_hr_threshold','num_picks','win_rate','total_profit','roi']]

both_under = under_profit.merge(under_test,on=['diff_threshold','pred_hr_threshold']).query('total_profit > 0 and train_profit > 0')
both_over = over_profit.merge(over_test,on=['diff_threshold','pred_hr_threshold']).query('total_profit > 0 and train_profit > 0')

best_under = best_under.append(both_under)
best_over = best_over.append(both_over)
#%%
final_ratings_test[['under_pick','over_pick']] = 0
for i in range(len(final_ratings_test)):
    if final_ratings_test['under_diff'][i] >= .045 and final_ratings_test.under[i] >= .84:
        final_ratings_test.under_pick[i] = 1
    if final_ratings_test.over[i] >= .22 and final_ratings_test['over_diff'][i] >= .015:
        final_ratings_test.over_pick[i] = 1
    continue

sum(final_ratings_test.over_pick)
sum(final_ratings_test.under_pick)

final_ratings_test['profit'] = 0
for i in range(len(final_ratings_test)):
    if final_ratings_test.under_pick[i] == 0 and final_ratings_test.over_pick[i] == 0:
        continue
    elif final_ratings_test.under_pick[i] == 1 and final_ratings_test.hr[i] == 1:
        final_ratings_test.profit[i] = -10
    elif final_ratings_test.over_pick[i] == 1 and final_ratings_test.hr[i] == 0:
        final_ratings_test.profit[i] = -10
    else:
        final_ratings_test.profit[i] = (round(((100/final_ratings_test.under[i])/10)-10,2)*final_ratings_test.under_pick[i])+(round(((100/final_ratings_test.over[i])/10)-10,2)*final_ratings_test.over_pick[i])

sum(final_ratings_test.profit)/((sum(final_ratings_test.over_pick)+sum(final_ratings_test.under_pick))/10)


final_ratings.to_csv('historical_profit_calc.csv',index=False)
# RECALC RATINGS - PRED_HR_ODDS



test_df_2 = final_ratings.iloc[X_te.index].reset_index(drop=True)

test_df_2[['under_pick','over_pick']] = 0
for i in range(len(test_df_2)):
    if test_df_2.under[i] >= .84 and test_df_2['under_diff'][i] >= .03:
        test_df_2.under_pick[i] = 1
    elif test_df_2.over[i] >= .12 and test_df_2['over_diff'][i] >= .04:
        test_df_2.over_pick[i] = 1
        continue
    
    
test_df_2['profit'] = 0
for i in range(len(test_df_2)):
    if test_df_2.under_pick[i] == 0 and test_df_2.over_pick[i] == 0:
        continue
    elif test_df_2.under_pick[i] == 1 and test_df_2.hr[i] == 1:
        test_df_2.profit[i] = -10
    elif test_df_2.over_pick[i] == 1 and test_df_2.hr[i] == 0:
        test_df_2.profit[i] = -10
    else:
        test_df_2.profit[i] = (round(((100/test_df_2.under[i])/10)-10,2)*test_df_2.under_pick[i])+(round(((100/test_df_2.over[i])/10)-10,2)*test_df_2.over_pick[i])
#%%
final_ratings_train[['under_pick','over_pick']] = 0
for i in range(len(final_ratings_train)):
    if final_ratings_train.under[i] >= .84 and final_ratings_train['under_diff'][i] >= .045:
        final_ratings_train.under_pick[i] = 1
    elif final_ratings_train.over[i] >= .22 and final_ratings_train['over_diff'][i] >= .015:
        final_ratings_train.over_pick[i] = 1
    else:
        continue

sum(final_ratings_train.over_pick)
sum(final_ratings_train.under_pick)

final_ratings_train['profit'] = 0
for i in range(len(final_ratings_train)):
    if final_ratings_train.under_pick[i] == 0 and final_ratings_train.over_pick[i] == 0:
        continue
    elif final_ratings_train.under_pick[i] == 1 and final_ratings_train.hr[i] == 1:
        final_ratings_train.profit[i] = -10
    elif final_ratings_train.over_pick[i] == 1 and final_ratings_train.hr[i] == 0:
        final_ratings_train.profit[i] = -10
    else:
        final_ratings_train.profit[i] = (round(((100/final_ratings_train.under[i])/10)-10,2)*final_ratings_train.under_pick[i])+(round(((100/final_ratings_train.over[i])/10)-10,2)*final_ratings_train.over_pick[i])

sum(final_ratings_train.profit)/((sum(final_ratings_train.over_pick)+sum(final_ratings_train.under_pick))/10)

test = final_ratings_test.groupby('year').agg(
    profit=('profit','sum'),
    picks=('over_pick','sum'))
    #o_picks=('under_pick','sum'))
#test['picks'] = test.u_picks + test.o_picks
test['roi'] = (test.profit/(test.picks*10))*100
#test = test.drop(columns=['o_picks','u_picks']).reset_index()

final_ratings_test[['under_pick','over_pick']] = 0
for i in range(len(final_ratings_test)):
    if final_ratings_test['under_diff'][i] >= .035 and final_ratings_test.under[i] >= .65:
        final_ratings_test.under_pick[i] = 1
    elif final_ratings_test.over[i] >= .18 and final_ratings_test['over_diff'][i] >= .02:
        final_ratings_test.over_pick[i] = 1
    continue

sum(final_ratings_test.over_pick)
sum(final_ratings_test.under_pick)

final_ratings_test['profit'] = 0
for i in range(len(final_ratings_test)):
    if final_ratings_test.under_pick[i] == 0 and final_ratings_test.over_pick[i] == 0:
        continue
    elif final_ratings_test.under_pick[i] == 1 and final_ratings_test.hr[i] == 1:
        final_ratings_test.profit[i] = -10
    elif final_ratings_test.over_pick[i] == 1 and final_ratings_test.hr[i] == 0:
        final_ratings_test.profit[i] = -10
    else:
        final_ratings_test.profit[i] = (round(((100/final_ratings_test.under[i])/10)-10,2)*final_ratings_test.under_pick[i])+(round(((100/final_ratings_test.over[i])/10)-10,2)*final_ratings_test.over_pick[i])

sum(final_ratings_test.profit)/((sum(final_ratings_test.over_pick)+sum(final_ratings_test.under_pick))/10)

#%% DOCUMENTATION OF FINAL RESULTS

# OVERS
"""
Rating Weight: 25%
Diff: 2%
Threshold: 20%

Validation Set(60%): 351 Picks, 10.52% Profit
Test Set: 228 Picks, 4.81% Profit
Total: 579 Picks, 8.27% Profit

Breakdown by year:
    2023: 185 Picks, 8.70% Profit
    2024: 213 Picks, 8.91% Profit
    2025: 113 Picks, 9.4% Profit
"""

# Unders
"""
Rating Weight: 15%
Diff: 4.5%
Threshold: 84%

Validation Set(40%): 339 Picks, 2.06% Profit
Test Set: 520 Picks, 3.65% Profit
Total: 859 Picks, 3.02% Profit

Breakdown by year:
    2023: 130 Picks, 1.17% Profit
    2024: 310 Picks, 3.94% Profit
    2025: 419 Picks, 2.9% Profit
"""

# TOTALS
"""
Overs:
    Prorated 220-230 Picks per year, 8.9% profit
Unders:
    Prorated 345-370 Picks per year, 3.03% Profit

TOTAL:
    550-600 Picks per year, 5.2% Profit
    
VERY EXCITING!!

"""


"""
Method:
    divide rating in half, create under and over ratings seperately
    haven't added things like weather or pullfb%
    
"""