# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 19:06:18 2025

@author: Brendan
"""
#%% Section 1: collect data

"""
In this section, we collect all the pitches seen/thrown since the start of 2025.
To do this, we get a list of all hitters/pitchers that have at least 10 PA/BF
over the season, in order to still project players with very limited sample sizes.

Due to this method of data collection taking a long time, we will only be 
updating player's stats once a week"""

import pybaseball as bb
import pandas as pd
from numpy import nan

hitters = pd.read_csv('all_batters.csv')
hitters = hitters.rename(columns={'last_name, first_name':'name'})
hitters = hitters.drop_duplicates(subset=['name','player_id'], keep='first').reset_index(drop=True)

for i in range(0,len(hitters)):
    stats = bb.statcast_batter("2024-03-01","2025-11-01",hitters.iloc[:,1][i])
    stats = stats.query('game_type == "R"').dropna(subset='pitch_type')
    stats = stats[~stats['pitch_type'].isin(['SC','PO','CS','FA','EP',nan,'AB','FC','IN'])]
    stats = stats[['game_year','player_name','pitch_type','release_speed','events','description','stand','p_throws',
             'bb_type','zone','launch_speed','spin_axis','launch_angle','release_spin_rate','estimated_woba_using_speedangle','hit_distance_sc',
             'launch_speed_angle','attack_angle','attack_direction','swing_path_tilt','age_bat','age_pit','bat_speed','swing_length','hit_location']]
    stats['playerid'] = hitters.iloc[:,1][i]
    if i == 0:
        pitches_b = stats
    else:
        pitches_b = pitches_b.append(stats)
pitches_b = pitches_b.reset_index().drop(columns='index').drop_duplicates()
pitches_b.to_csv('batters.csv',index=False)
del stats, i
#%%
import pybaseball as bb
import pandas as pd
from numpy import nan
pitchers = pd.read_csv('all_pitchers.csv')
pitchers = pitchers.rename(columns={'last_name, first_name':'name'})
pitchers = pitchers.drop_duplicates(subset=['name','player_id'], keep='first').reset_index(drop=True)
for i in range(0,len(pitchers)):
    stats = bb.statcast_pitcher("2024-03-01","2025-11-01",pitchers.iloc[:,1][i])
    if stats.empty:
        continue
    stats = stats.query('game_type == "R"').dropna(subset='pitch_type')
    stats = stats[~stats['pitch_type'].isin(['SC','PO','CS','FA','EP',nan,'AB','FC','IN'])]
    stats = stats[['game_year','game_date','player_name','pitch_type','release_speed','events','description','stand','p_throws','release_pos_z','release_pos_x','plate_x','plate_z',
             'bb_type','zone','launch_speed','spin_axis','launch_angle','release_spin_rate','release_extension','estimated_woba_using_speedangle','hit_distance_sc','n_priorpa_thisgame_player_at_bat',
             'launch_speed_angle','attack_angle','attack_direction','swing_path_tilt','age_bat','age_pit','bat_speed','swing_length','at_bat_number','arm_angle','n_thruorder_pitcher','outs_when_up','inning','batter']]
    stats['playerid'] = pitchers.iloc[:,1][i]
    if i == 0: pitches = stats
    else: pitches = pitches.append(stats)
    
pitches = pitches.reset_index().drop(columns='index').drop_duplicates()
pitches.to_csv('pitchers.csv',index=False)
del stats, i
#%% altering datasets

"""
After collecting data, we must add/edit some columns so it matches with the 2024
data, this is not included in the UDF because this only needs to run every time
new data is collected, and will save time"""

# pitchers
import pandas as pd
from unidecode import unidecode
from numpy import select,nan,inf
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
conditions = [pitchers['launch_angle'].isna(),
pitchers['launch_angle'] < 10,
(pitchers['launch_angle'] >= 10) & (pitchers['launch_angle'] < 25),
(pitchers['launch_angle'] >= 25) & (pitchers['launch_angle'] < 50),
pitchers['launch_angle'] >= 50]
choices = ['nan','ground_ball','line_drive','fly_ball','popup']
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


# batters
batters = pd.read_csv('batters.csv')

# Removing bunts from the analysis, would screw up ev and la metrics
batters = batters[~(batters['description'].str.contains('bunt', case=False))]
#remove intent_ball, velo would be messed up
batters = batters[~(batters['description'].str.contains('intent_ball', case=False))]
# distinction between blocked or not and foul_tip/swinging_strike doesn't matter
batters['description'] = batters['description'].str.replace('swinging_strike_blocked', 'swinging_strike')
batters['description'] = batters['description'].str.replace('blocked_ball', 'ball')
batters['description'] = batters['description'].str.replace('foul_tip', 'swinging_strike') 

# discovered that bb_types were not correct so changed them manually

batters = batters.reset_index().drop(columns='index')
conditions = [batters['launch_angle'].isna(),
batters['launch_angle'] < 10,
(batters['launch_angle'] >= 10) & (batters['launch_angle'] < 25),
(batters['launch_angle'] >= 25) & (batters['launch_angle'] < 50),
batters['launch_angle'] >= 50]
choices = ['nan','ground_ball','line_drive','fly_ball','popup']
batters['bb_type'] = select(conditions, choices) 
batters['bb_type'].replace('nan',batters['launch_angle'][1], inplace = True)
batters = batters[~batters['events'].isin(['sac_bunt','sac_bunt_double_play'])]
batters = batters.reset_index().drop(columns='index')

batters['in_play'] = (batters['description'] == 'hit_into_play').astype(int)
batters['barrel'] = (batters['launch_speed_angle'] == 6).astype(int)
batters['weak'] = (batters['launch_speed_angle'].isin([1,2])).astype(int)
batters['fly_ball'] = batters.apply(lambda row: 1 if row['bb_type'] == 'fly_ball' and row['description'] != 'foul' else 0, axis = 1)
batters['ground_ball'] = batters.apply(lambda row: 1 if row['bb_type'] == 'ground_ball' and row['description'] != 'foul' else 0, axis = 1)
batters['line_drive'] = batters.apply(lambda row: 1 if row['bb_type'] == 'line_drive' and row['description'] != 'foul' else 0, axis = 1)
batters['whiff'] = (batters['description'] == 'swinging_strike').astype(int)
batters['swing'] = (batters['description'].isin(['swinging_strike','hit_into_play','foul'])).astype(int)
batters['home_run'] = (batters['events'] == 'home_run').astype(int)
batters['hh'] = (batters['launch_speed'] >= 95).astype(int)
batters['in_zone'] = (batters['zone'] < 10).astype(int)
batters['chase'] = batters.apply(lambda row: 1 if row['swing'] == 1 and row['in_zone'] == 0 else 0, axis = 1)


del choices, conditions

from modify_batters import modify_batters
from modify_pitchers import modify_pitchers

batters['player_name'] = batters['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
old_hits = pd.read_csv('old_hits.csv')
hit_stats = modify_batters(batters, old_hits)

pitchers['player_name'] = pitchers['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
old_pitch = pd.read_csv('old_pitch.csv')
pitchers = pitchers.drop_duplicates()
pitch_stats = modify_pitchers(pitchers, old_pitch)

pitch_stats = pitch_stats.rename(columns={'pitch_count':'count'})
pitch_stats = pitch_stats.rename(columns={'pitch_type':'pitch'})
hit_stats = hit_stats.rename(columns={'pitch_count':'count'})
hit_stats = hit_stats.rename(columns={'pitch_type':'pitch'})

hit_stats.to_csv('hit_stats.csv',index=False)
pitch_stats.to_csv('pitch_stats.csv',index=False)
#%% getting results from yesterday's games
import pandas as pd
import pybaseball as bb
yesterday = pd.read_csv('daily_lineups.csv')
yesterday['hr'] = 0
for i in range(0,len(yesterday)):
    name = yesterday.playerid[i]
    date = str(yesterday.date[i])
    stats = bb.statcast_batter(date,date,name)
    if stats.empty: yesterday = yesterday.drop(i)
    elif 'home_run' in list(stats.events.unique()): yesterday.hr[i] = 1
    else: continue
yesterday = yesterday.reset_index(drop=True)

yesterday['profit'] = 0
for i in range(len(yesterday)):
    if yesterday.under_pick[i] == 0 and yesterday.over_pick[i] == 0:
        continue
    elif yesterday.under_pick[i] == 1 and yesterday.hr[i] == 1:
        yesterday.profit[i] = -10
    elif yesterday.over_pick[i] == 1 and yesterday.hr[i] == 0:
        yesterday.profit[i] = -10
    else:
        yesterday.profit[i] = (round(((100/yesterday.Under[i])/10)-10,2)*yesterday.under_pick[i])+(round(((100/yesterday.Over[i])/10)-10,2)*yesterday.over_pick[i])
archive = pd.read_excel('archive.xlsx')
archive = archive.append(yesterday)
archive.to_excel('new_archive.xlsx',index=False)
del yesterday, name, date, stats, i

sum(archive.profit)/(sum(archive.under_pick)*10)
#%% rosters using r code
"""
next we have to pull every team's roster for the day. Unfortunately, this can 
only be done with baseballr to my knowledge, so with the help of Claude, I
imported my r code for pulling rosters and put it into python."""
import pandas as pd
archive = pd.read_excel('archive.xlsx')
import subprocess
import tempfile
import os
import pandas as pd
from io import StringIO
from unidecode import unidecode

r_code = """
library(baseballr)
team_ids <- c(108:121, 133:147, 158)
all_rosters <- lapply(team_ids, function(x) { roster <- try(mlb_rosters(team_id = x, season = 2025, roster_type = 'active'), silent = TRUE); if(!inherits(roster, "try-error")) { roster$team_id <- x; roster } })
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
del f,r_code,result,temp_script
#% loading datasets

"""
now we run our UDF's for collecting the stats we need to predict HR chance.
"""

import pandas as pd
from unidecode import unidecode
from datetime import datetime
start = datetime.now()
hit_stats = pd.read_csv('hit_stats.csv')
pitch_stats = pd.read_csv('pitch_stats.csv')
players['person_full_name'] = players['person_full_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
ids = pd.read_excel('HR_factors.xlsx')
players['team_id'] = players['team_id'].astype(int)
players = players.merge(ids, on='team_id', how='left')
players = players[['person_id','Stadium','person_full_name']]
players = players.rename(columns={'person_id':'playerid','person_full_name':'player_name'})
players['playerid'] = players['playerid'].astype(int)
hit_stats = hit_stats.merge(players,how='inner',on=['playerid','player_name'])
hit_stats = hit_stats.drop_duplicates().reset_index(drop=True)
pitch_stats = pitch_stats.merge(players,how='inner',on=['playerid','player_name'])
pitch_stats = pitch_stats.drop_duplicates()
pitch_stats = pitch_stats.reset_index(drop=True)
#% bullpens

"""
the biggest reason we pulled rosters was for this. I haven't gotten far enough
to be able to acurately predict which relief pitcher will come into the game,
so instead I am pulling every player on the team that has a low avg batters faced,
and combining their stats so as to get an aggregate bullpen performance
"""

teams = list(pitch_stats.Stadium.unique())
bullpen_stats = pd.DataFrame()
for team in teams:
    scope = pitch_stats.query('Stadium == @team')
    scope = scope.query('avg_bf < 7.0 and std_bf < 4.0')
    pitch_list = list(scope.pitch.unique())
    for pitch in pitch_list:
        pbp = scope.query('pitch == @pitch')
        pbp = pbp.groupby('pitch').agg(
            pred_hr = ('pred_hr','mean'),
            pitch_count = ('count','sum')).reset_index().round(2)
        pbp['Stadium'] = team
        bullpen_stats = bullpen_stats.append(pbp)
    bullpen_stats = bullpen_stats.reset_index(drop=True)
    scope = bullpen_stats.query('Stadium == @team')
    total_count = scope['pitch_count'].sum()
    bullpen_stats.loc[bullpen_stats['Stadium'] == team, 'percentage'] = bullpen_stats.loc[bullpen_stats['Stadium'] == team, 'pitch_count'] / total_count

bullpen_stats = bullpen_stats.round(3)

del teams, scope, pbp, total_count, pitch_list, pitch, team

#% get lineup data via rotowire

"""
self-explanatory - we go to rotowire and pull their projected lineups for the day
"""

import datetime as dt
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Firefox, FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from unidecode import unidecode
from random import randint
#driver.close()
#options = Options()
opts = FirefoxOptions()
opts.add_argument("--width=950")
opts.add_argument("--height=1025")
driver_path = "C:\\Users\\brend\\Downloads\\geckodriver-v0.33.0-win-aarch64(1).zip\\geckodriver.exe"
driver = Firefox(options=opts)
#driver = webdriver.Chrome(chrome_options = options, executable_path = driver_path)
url = "https://www.rotowire.com/baseball/daily-lineups.php"
driver.get(url)
time.sleep(7)
lineups = pd.DataFrame()
g = 1
while g < 30:
    try:
         a_team = driver.find_element(By.CSS_SELECTOR, 'div.lineup:nth-child('+str(g)+') > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2)').text
    except NoSuchElementException:
        g += 1
        continue
    if driver.find_element(By.CSS_SELECTOR, 'div.lineup:nth-child('+str(g)+') > div:nth-child(2) > div:nth-child(1)').text == 'Daily Fantasy MLB Tools' :
        break
    else:
        pass
    h_team = driver.find_element(By.CSS_SELECTOR, 'div.lineup:nth-child('+str(g)+') > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2)').text
    for p in range(1,3): # needed for both home and away
        order = 1
        try:
            if len(driver.find_element(By.CSS_SELECTOR, 'div.lineup:nth-child('+str(g)+') > div:nth-child(2) > div:nth-child(3) > ul:nth-child('+str(p)+') > li:nth-child(4)').text) == 0:
                x = range(3,20,2)
            else:
                x = range(3,12)
        except NoSuchElementException:
            break
        for i in x:
            if p == 1:
                pi = 2
                status = 'away'
            else:
                pi = 1
                status = 'home'
            pitcher = pd.DataFrame([driver.find_element(By.CSS_SELECTOR, 'div.lineup:nth-child('+str(g)+') > div:nth-child(2) > div:nth-child(3) > ul:nth-child('+str(pi)+') > li:nth-child(1) > div:nth-child(1)').text],columns=['name'])
            pitcher['handedness'] = pitcher.iloc[0,0].split()[-1]
            pitcher['name'][0] = pitcher['name'][0][:-2]
            if '.' in pitcher['name'][0]:
                last_name = pitcher['name'][0].split()[-1]
                initial = pitcher['name'][0][0]
                name_check = pitch_stats[pitch_stats['player_name'].str.contains(last_name,case=False)]
                name_check = name_check[name_check['player_name'].str.startswith(initial)]
                if name_check.empty:
                    pass
                else:
                    pitcher['name'][0] = list(name_check['player_name'])[0]
            
            player = pd.DataFrame([driver.find_element(By.CSS_SELECTOR, 'div.lineup:nth-child('+str(g)+') > div:nth-child(2) > div:nth-child(3) > ul:nth-child('+str(p)+') > li:nth-child('+str(i)+')').text])
            player = player[0].str.split(' ', expand=True)
            player = player.rename(columns={0:'position',(player.shape[1]-1):'handedness'})
            player['name'] = player.iloc[:, 1:-1].apply(lambda x: ' '.join(x), axis=1)
            player = player.drop(columns=player.columns[1:-2])
            if '.' in player['name'][0]:
                last_name = player['name'][0].split()[-1]
                initial = player['name'][0][0]
                name_check = hit_stats[hit_stats['player_name'].str.contains(last_name,case=False)]
                name_check = name_check[name_check['player_name'].str.startswith(initial)]
                if name_check.empty:
                    pass
                else:
                    player['name'][0] = list(name_check['player_name'])[0]
                    
            player[['pitcher','p_throws']] = pitcher
            player[['lineup_spot','away_team','home_team','status']] = order,a_team,h_team,status
            lineups = lineups.append(player)
            order += 1
    g += 1
    lineups = lineups.reset_index(drop=True)    
lineups = lineups.dropna().reset_index(drop=True)
lineups['name'] = lineups['name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
lineups['pitcher'] = lineups['pitcher'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
driver.close()
del g,p,pi,a_team,h_team,driver,driver_path,opts,url,x,status,pitcher,last_name,initial,name_check,player,order

#% ratings for matchups



from statistics import mean
from scipy import stats
from math import floor,ceil
from scipy import stats

# average PA count for each lineups position
pa_per_game = {1: 4.65,2: 4.55,3: 4.43,4: 4.33,5: 4.24,6: 4.13,7: 4.01,8: 3.90,9: 3.77}
# modifiers based on predicted plate apperance totals
pa_mod = {0:0,1:0.44,2:1.21,3:1.50,4:1.50,5:1.5}
lineups['rating'] = 0
lineups['team'] = 'none'
for i in range(0,len(lineups)):
    
    # getting basic info - teams, lineup spots, and making sure we have data on all players
    
    spot = lineups.lineup_spot[i]
    pa_count = pa_per_game[spot] 
    b_name = lineups.name[i]
    p_name = lineups.pitcher[i]
    stadium = lineups.home_team[i]
    if lineups.status[i] == 'home':
        opp_bp = lineups.away_team[i]
        lineups.team[i] = lineups.home_team[i]
        bat = hit_stats.query('Stadium == @stadium')
    else:
        opp_bp = lineups.home_team[i]
        lineups.team[i] = lineups.away_team[i]
        bat = hit_stats.query('Stadium == @lineups.away_team['+str(i)+']')
    bat = bat[bat['player_name'].str.contains(b_name,case=False)].reset_index(drop=True)
    pitch = pitch_stats[pitch_stats['player_name'].str.contains(p_name,case=False)].reset_index(drop=True)
    bp = bullpen_stats.query('Stadium == @opp_bp').reset_index(drop=True)


    """
    after making sure that both players have data, we compare their ML-predicted
    home run ratings, and created a weighted rating based on how often the pitcher
    throws that pitch. 

    """
    if not bat.empty and not pitch.empty:
        matchup = bat.merge(pitch,how='outer',on='pitch').fillna(0)
        matchup['rating'] = 0
        matchup = matchup.sort_values(by='pred_hr_x',ascending = False)
        matchup = matchup.drop_duplicates(subset='pitch',keep='first').reset_index(drop=True)
        matchup['percentage'] = round(matchup['count_y']/sum(matchup['count_y']),3)
        for q in range(0,len(matchup)):
            bat_r = matchup['pred_hr_x'][q]
            pit_r = matchup['pred_hr_y'][q]
            amt = matchup['percentage'][q]
            if bat_r == 0:
                real_pitches = matchup.query('pred_hr_x > 0')
                matchup['rating'][q] = round((pit_r+(mean(real_pitches.pred_hr_x)*0.9))*amt,2)
            elif pit_r == 0:
                continue
            else:
                matchup['rating'][q] = round((pit_r++bat_r)*amt,2)
        rating = round(sum(matchup.rating),2)
    else: continue
        
# prob for each player to face starter
    """
    once the rating vs the starter is calculated, we then calculate the projected
    PA count for specifically starter vs hitter.
    """


    bf_mean = pitch.avg_bf[0]
    bf_std = pitch.std_bf[0]
    if bf_std != bf_std:
        bf_std = round(mean(pitch_stats.query('avg_bf >= 10.0')['std_bf'].dropna()),2)
    else:
        pass
    starter_pa = 0
    pa = 1
    for p in range(spot,(spot+37),9):
        zs1 = round((p-bf_mean)/bf_std,2)
        zs2 = round(((p+9)-bf_mean)/bf_std,2)        
        prob = round(stats.norm.cdf(zs2) - stats.norm.cdf(zs1),5)
        starter_pa += (pa*prob)
        pa += 1
    starter_pa = starter_pa.round(2)
    bp_pa = pa_count - starter_pa

    # modifiers for plate apperances, handedness and stadium
    
    bats = lineups.handedness[i]
    throws = lineups.p_throws[i]
    if bats == "S" and throws == 'R':
        bats = 'L'
    elif bats == "S" and throws == 'L':
        bats = 'R'
    park = ids.query('Handedness == @bats and Stadium == @stadium').reset_index(drop=True)
    rating = round(rating*park.HR[0],2)
    if bats == throws:
        rating = rating + matchup['plat_disc'][0]/2
    else:
        rating = rating + (matchup['plat_disc'][0]/2*(-1))

    starter_mod = ((ceil(starter_pa)-starter_pa).round(2)*pa_mod[floor(starter_pa)]) + ((starter_pa-floor(starter_pa)).round(2)*pa_mod[ceil(starter_pa)])
    rating = round(rating*starter_mod,2)

# now time for bullpen
    """
    now the same process is repeated, base rating, PA count, and other modifiers
    for the hitters vs the aggregate bullpen"""
    matchup = bat.merge(bp,how='outer',on='pitch').fillna(0)
    matchup['rating'] = 0
    for q in range(0,len(matchup)):
        bat_r = matchup['pred_hr_x'][q]
        pit_r = matchup['pred_hr_y'][q]
        amt = matchup['percentage'][q]
        if bat_r == 0:
            real_pitches = matchup.query('pred_hr_x > 0')
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
    lineups['rating'][i] = hr_rating    
lineups = lineups.rename(columns={'name':'player'})
merger = hit_stats[['player_name','playerid','Stadium']]
merger = merger.rename(columns={'Stadium':'team','player_name':'player'}).drop_duplicates()
lineups = lineups.merge(merger, how='left',on = ['team','player'])
del b_name, p_name, matchup, amt, rating, bat, pitch, bat_r, pit_r, zs1, zs2, bp_pa, bf_mean, bf_std, prob, pa, bp, q, i, bp_rating, real_pitches, spot, pa_per_game, pa_mod, starter_mod,bp_mod,stadium,opp_bp,p,hr_rating,pa_count,starter_pa,park,bats,throws,merger

#% player odds


"""
pulling player HR odds from popular sportsbooks: in this case just fanduel and
draftkings"""

games = int(len(lineups.query('lineup_spot == 1'))/2)
#games = 1
opts = FirefoxOptions()
opts.add_argument("--width=1350")
opts.add_argument("--height=1025")
driver_path = "C:\\Users\\brend\\Downloads\\geckodriver-v0.33.0-win-aarch64(1).zip\\geckodriver.exe"
driver = Firefox(options=opts)
#driver = webdriver.Chrome(chrome_options = options, executable_path = driver_path)
url = "https://crazyninjaodds.com/site/browse/games.aspx"
driver.get(url)
time.sleep(3)
driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_WebUserControl_FilterSport_DropDownListSport > option:nth-child(2)').click()
driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_WebUserControl_FilterLeague_DropDownListLeague > option:nth-child(2)').click()
timer = driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_WebUserControl_FilterMaximumDateHours_TextBoxMaximumDateHours')
timer.send_keys(15)
driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_ButtonUpdate').click()
all_odds = pd.DataFrame()
for i in range(1,games+1):
    time.sleep(2)
    link = driver.find_element(By.XPATH, '/html/body/form/div[3]/layout-body-main/layout-body-main-right/div[2]/div[2]/table/tbody/tr['+str(i)+']/td[1]/a').text
    if 'G2' in link:
        continue
    else:
        link = driver.find_element(By.XPATH, '/html/body/form/div[3]/layout-body-main/layout-body-main-right/div[2]/div[2]/table/tbody/tr['+str(i)+']/td[1]/a')
    driver.execute_script("arguments[0].removeAttribute('target')", link)
    link.click()
    time.sleep(2)
    for q in range(1,110):
        event = driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListMarket > option:nth-child('+str(q)+')').text
        if event == 'Player Home Runs':
            driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListMarket > option:nth-child('+str(q)+')').click()
            break
        else:
            continue
    time.sleep(2)
    tab = driver.find_element(By.TAG_NAME, 'table')
    tab_html = tab.get_attribute('outerHTML')
    df = pd.read_html(tab_html)[0]
    all_odds = all_odds.append([df])
    driver.back()
all_odds = all_odds.reset_index(drop=True)
driver.close()
del games, opts, driver_path, driver, event,timer,url,link,tab,tab_html,df,q,i

# fixing df and getting odd diffs
all_odds = all_odds[['Bet Name','MGM','ESPN']].reset_index()
split_df = all_odds['Bet Name'].str.split(' ', expand=True)
try: split_df.columns = ['0', '1', '2','3','4','5']
except ValueError:split_df.columns = ['0', '1', '2','3','4']
split_df[['player','bet','amount']] = 'zero'
for i in range(1,len(all_odds)):
    if split_df['3'][i] == None:
        continue
    if split_df['2'][i] in ['Over','Under']:
        split_df['player'][i] = split_df['0'][i]+' '+split_df['1'][i]
        split_df['bet'][i] = split_df['2'][i]
        split_df['amount'][i] = split_df['3'][i]
    elif split_df['3'][i] in ['Over','Under']:
        split_df['player'][i] = split_df['0'][i]+' '+split_df['1'][i]+' '+split_df['2'][i]
        split_df['bet'][i] = split_df['3'][i]
        split_df['amount'][i] = split_df['4'][i]
    else:
        split_df['player'][i] = split_df['0'][i]+' '+split_df['1'][i]+' '+split_df['2'][i]+' '+split_df['3'][i]
        split_df['bet'][i] = split_df['4'][i]
        split_df['amount'][i] = split_df['5'][i]
split_df[['MGM','ESPN']] = all_odds[['MGM','ESPN']]
try: split_df = split_df.drop(columns=['0','1','2','3','4','5'])
except KeyError: split_df = split_df.drop(columns=['0','1','2','3','4'])
split_df = split_df.query('amount == "0.5"')

split_df = split_df.dropna(subset=['MGM','ESPN'],how='all')
split_df = split_df[['player','bet','MGM','ESPN']]
melted = split_df.melt(id_vars=['player', 'bet'], value_vars=['MGM', 'ESPN'], var_name='book', value_name='odds')
max_odds = melted.groupby(['player', 'bet'])['odds'].max().reset_index()

max_books = melted.loc[melted.groupby(['player', 'bet'])['odds'].idxmax()][['player', 'bet', 'book']].rename(columns={'book': 'best_book'})

result = max_odds.pivot(index='player', columns='bet', values='odds').reset_index()
book_result = max_books.pivot(index='player', columns='bet', values='best_book').reset_index()
book_result = book_result.rename(columns={'Over': 'over_book', 'Under': 'under_book'})
result = result.merge(book_result, on='player')
result = result.drop_duplicates(subset='player',keep='first')
result['player'] = result['player'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
lineups = lineups.merge(result,on='player')
lineups = lineups.query('rating != 0').reset_index(drop=True)
del split_df,all_odds,i,result,book_result,max_odds,melted


end = datetime.now()
time = end-start
del start,end

"""
after collecting sufficient data, we found that the mean rating was around 5.6, and 
every 1 in 7 batters produced a home run. So to calculate odds, we are claiming that
any hitter with a 5.4 rating has a 14.4% chance to hit a homerun (+800), and any
decimal point either above or below that will act as a modifier to the percentage.

For example, if the matchup rating was 10.0, that player would have a 28.6%, or
+250, chance of hitting a home run, since 5 is double 10"""
from sklearn.linear_model import LogisticRegression
X = archive.rating.dropna()
y = archive.hr.drop([2991,2992,2993])
result = LogisticRegression().fit(X.values.reshape(-1,1),y)
pred = lineups.rating
lineups[['pred_under','pred_over']] = (result.predict_proba(pred.values.reshape(-1,1))).round(3)
lineups.Over = ((100/(lineups.Over+100))).round(3)
lineups.Under = ((lineups.Under/(lineups.Under-100))).round(3)
lineups['under_diff'] = lineups.pred_under-lineups.Under
lineups['over_diff'] =lineups.pred_over-lineups.Over
        



"""
after calculating our own odds, we will compare them to the more favorable of the
sportsbooks we pulled, and produce the difference in our odds vs theirs, in this
case, a higher number would be seen as more percieved value over the sports books"""


""" Next is the criteria for selecting our "pick" for that day's games. After
running ROI analysis on the data collected throughout July, there are two boxes
that need to be checked in order to bet a player's under:
    
    - A sportsbook predicted chance of no HR at 88% or higher
    - a positive difference between predicted odds and sportsbook odds of
      at least 150
     
    Using this criteria, we have a track record of 64-2, with an 8.1% ROI
"""

lineups[['under_pick','over_pick']] = 0
for i in range(len(lineups)):
    if lineups.Under[i] >= .820 and lineups['under_diff'][i] >= .035:
        lineups.under_pick[i] = 1
    elif lineups.Over[i] >= .080 and lineups['over_diff'][i] >= .055:
        lineups.over_pick[i] = 1
        continue
from datetime import date
lineups['date'] = date.today()

lineups.to_csv('daily_lineups.csv',index=False)
#%% aggregate ratings
import pandas as pd
pitch_stats = pd.read_csv('pitch_stats.csv')
names = pitch_stats[['player_name','playerid','bip','avg_bf']]
names = names.drop_duplicates(subset=['player_name','playerid'], keep='first').reset_index(drop=True)
names = names.query('avg_bf > 8').reset_index(drop=True)
names['pred_hr'] = 0
for i in range(len(names)):
    name = names['playerid'][i]
    player = pitch_stats.query('playerid == @name').reset_index().drop(columns='index')
    total = 0
    for q in range(len(player)):
        total += player.pred_hr[q]*(player.bip[q]/sum(player.bip))
    names.bip[i] = sum(player.bip)
    names.pred_hr[i] = total.round(2)
#
total = 0
for i in range(len(names)):
    total += names.pred_hr[i] * (names.bip[i]/sum(names.bip))