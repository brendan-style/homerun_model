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
hitters = pd.read_csv('all_batters.csv')
hitters = hitters.rename(columns={'last_name, first_name':'name'})
hitters = hitters.drop_duplicates(subset=['player_id','name']).reset_index(drop=True)
for i in range(0,len(hitters)):
    stats = bb.statcast_batter("2025-03-01","2025-11-01",hitters.iloc[:,1][i])
    stats = stats.query('game_type == "R"').dropna(subset='pitch_type')
    stats = stats[['game_date','player_name','pitch_type','release_speed','events','description','stand','p_throws',
             'bb_type','zone','launch_speed','spin_axis','launch_angle','release_spin_rate','estimated_woba_using_speedangle',
             'launch_speed_angle']]
    stats['hitter'] = hitters.iloc[:,1][i]
    stats['year'] = hitters.iloc[:,2][i]
    if i == 0:
        pitches_b = stats
    else:
        pitches_b = pitches_b.append(stats)
pitches_b = pitches_b.reset_index().drop(columns='index')
pitches_b.to_csv('2025_batters.csv',index=False)
del stats, i
from numpy import nan
pitchers = pd.read_csv('all_pitchers.csv')
pitchers = pitchers.rename(columns={'last_name, first_name':'name'})
pitchers = pitchers.drop_duplicates(subset=['player_id','name']).reset_index(drop=True)
for i in range(0,len(pitchers)):
    stats = bb.statcast_pitcher("2025-03-01","2025-11-01",pitchers.player_id[i])
    stats = stats.query('game_type == "R"').dropna(subset='pitch_type')
    stats = stats[~stats['pitch_type'].isin(['SC','PO','CS','FA','EP',nan,'AB','FC','IN'])]
    stats = stats[['player_name','pitcher','game_date','game_year','pitch_type','release_speed','events','description','stand','p_throws',
             'bb_type','launch_speed','launch_angle','release_spin_rate','estimated_woba_using_speedangle',
             'launch_speed_angle','pfx_x','pfx_z','release_extension','release_pos_x','release_pos_z','inning','outs_when_up','batter']]
    if i == 0:
      pitches_p = stats
    else:
        pitches_p = pitches_p.append(stats)
pitches_p = pitches_p.reset_index().drop(columns='index')
pitches_p.to_csv('2025_pitchers.csv',index=False)
del stats, i
#%% altering datasets

"""
After collecting data, we must add/edit some columns so it matches with the 2024
data, this is not included in the UDF because this only needs to run every time
new data is collected, and will save time"""

# pitchers
import pandas as pd
from numpy import select
from unidecode import unidecode
from numpy import nan
pitchers = pd.read_csv('2025_pitchers.csv')
pitchers = pitchers.rename(columns={'game_year':'year'})
pitchers['splits'] = pitchers.apply(lambda row: 'plat_disadv' if row['stand'] == row['p_throws'] else 'plat_adv', axis=1)
event_list = list(pitchers['description'].unique())
desc_list = list(pitchers['events'].unique())
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
    (pitchers['launch_angle'] > 25) & (pitchers['launch_angle'] <= 50),
    pitchers['launch_angle'] > 50]
choices = ['nan','ground_ball','line_drive','fly_ball','popup']
pitchers['bb_type'] = select(conditions, choices) 
pitchers['bb_type'].replace('nan',pitchers['launch_angle'][1], inplace = True)
pitchers = pitchers.reset_index().drop(columns='index')
pitchers['bbe'] = 0
for i in range(0,len(pitchers)):
    if pitchers['description'][i] == 'hit_into_play':
        pitchers['bbe'][i] = 1
    elif pitchers['description'][i] == 'foul':
        if int(pitchers['bb_type'][i] == pitchers['bb_type'][i]) == 1:
            pitchers['bbe'][i] = 1
        else: continue
    else: continue
pitchers['barrel'] = (pitchers['launch_speed_angle'] == 6).astype(int)
pitchers['weak'] = pitchers.apply(lambda row: 1 if row['launch_speed_angle'] == 1 or row['launch_speed_angle'] == 2 else 0, axis=1)
pitchers['fly_ball'] = (pitchers.apply(lambda row: 1 if row['bb_type'] == 'fly_ball' or row['bb_type'] == 'popup' else 0, axis=1)).astype(int) 
pitchers['ground_ball'] = (pitchers['bb_type'] == 'ground_ball').astype(int)
pitchers['line_drive'] = (pitchers['bb_type'] == 'line_drive').astype(int)
pitchers['whiff'] = (pitchers['description'] == 'swinging_strike').astype(int)
pitchers['swing'] = pitchers.apply(lambda row: 1 if row['description'] == 'swinging_strike' or row['description'] == 'hit_into_play' or row['description'] == 'foul'else 0, axis=1)
pitchers['home_run'] = (pitchers['events'] == 'home_run').astype(int)
pitchers['hh'] = (pitchers['launch_speed'] > 95).astype(int)
pitchers['foul'] = pitchers.apply(lambda row: 1 if row['description'] == 'foul' and row['bb_type'] != row['bb_type'] else 0, axis=1)
pitchers.to_csv('2025_pitchers.csv',index=False)

# batters
batters = pd.read_csv('2025_batters.csv')
batters['year'] = pd.to_datetime(batters['game_date']).dt.year
pitch_list = list(batters['pitch_type'].unique())
# first we remove non-valid pitch types or pitches wihtout much data
batters = batters[~batters['pitch_type'].isin(['SC','PO','CS','FA','EP',nan,'AB','FC','IN'])]
pitch_list = list(batters['pitch_type'].unique())
event_list = list(batters['description'].unique())
# Removing bunts from the analysis, would screw up ev and la metrics
batters = batters[~(batters['description'].str.contains('bunt', case=False))]
#remove intent_ball, velo would be messed up
batters = batters[~(batters['description'].str.contains('intent_ball', case=False))]
# distinction between blocked or not and foul_tip/swinging_strike doesn't matter
batters['description'] = batters['description'].str.replace('swinging_strike_blocked', 'swinging_strike')
batters['description'] = batters['description'].str.replace('blocked_ball', 'ball')
batters['description'] = batters['description'].str.replace('foul_tip', 'swinging_strike') 
event_list = list(batters['description'].unique())

# discovered that bb_types were not correct so changed them manually

batters = batters.reset_index().drop(columns='index')
conditions = [
    batters['launch_angle'].isna(),
    batters['launch_angle'] < 10,
    (batters['launch_angle'] >= 10) & (batters['launch_angle'] <= 25),
    (batters['launch_angle'] > 25) & (batters['launch_angle'] <= 50),
    batters['launch_angle'] > 50]
choices = ['nan','ground_ball','line_drive','fly_ball','popup']
batters['bb_type'] = select(conditions, choices) 
batters['bb_type'].replace('nan',batters['launch_angle'][1], inplace = True)
batters = batters[~batters['events'].isin(['sac_bunt','sac_bunt_double_play'])]
batters = batters.reset_index().drop(columns='index')
batters['bbe'] = 0
for i in range(0,len(batters)):
    if batters['description'][i] == 'hit_into_play':
        batters['bbe'][i] = 1
    elif batters['description'][i] == 'foul':
        if int(batters['bb_type'][i] == batters['bb_type'][i]) == 1:
            batters['bbe'][i] = 1
        else: continue
    else: continue
batters['barrel'] = (batters['launch_speed_angle'] == 6).astype(int)
batters['weak'] = batters.apply(lambda row: 1 if row['launch_speed_angle'] == 1 or row['launch_speed_angle'] == 2 else 0, axis=1)
batters['fly_ball'] = (batters.apply(lambda row: 1 if row['bb_type'] == 'fly_ball' or row['bb_type'] == 'popup' else 0, axis=1)).astype(int) 
batters['ground_ball'] = (batters['bb_type'] == 'ground_ball').astype(int)
batters['line_drive'] = (batters['bb_type'] == 'line_drive').astype(int)
batters['whiff'] = (batters['description'] == 'swinging_strike').astype(int)
batters['swing'] = batters.apply(lambda row: 1 if row['description'] == 'swinging_strike' or row['description'] == 'hit_into_play' or row['description'] == 'foul'else 0, axis=1)
batters['home_run'] = (batters['events'] == 'home_run').astype(int)
batters['hh'] = (batters['launch_speed'] > 95).astype(int)
batters['foul'] = batters.apply(lambda row: 1 if row['description'] == 'foul' and row['bb_type'] != row['bb_type'] else 0, axis=1)
batters.to_csv('2025_batters.csv',index=False)

del choices, conditions, desc_list, pitch_list, event_list, i

#%% rosters using r code

"""
next we have to pull every team's roster for the day. Unfortunately, this can 
only be done with baseballr to my knowledge, so with the help of Claude, I
imported my r code for pulling rosters and put it into python."""

import subprocess
import tempfile
import os
import pandas as pd
from io import StringIO

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
                       capture_output=True, text=True, check=True)
os.unlink(temp_script)
players = pd.read_csv(StringIO(result.stdout))
del f,r_code,result,temp_script
#%% loading datasets

"""
now we run our UDF's for collecting the stats we need to predict HR chance.
"""

from modify_batters import modify_batters
from modify_pitchers import modify_pitchers
import pandas as pd
from unidecode import unidecode
from datetime import datetime
start = datetime.now()
players['person_full_name'] = players['person_full_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
ids = pd.read_excel('HR_factors.xlsx')
batters_24 = pd.read_csv('2024_batters.csv')
batters_25 = pd.read_csv('2025_batters.csv')
batters = pd.concat([batters_24,batters_25])
del batters_24, batters_25
batters['player_name'] = batters['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
old_hits = pd.read_csv('final_rates_b.csv')
hit_stats = modify_batters(batters, old_hits,players,ids)
pitchers_24 = pd.read_csv('2024_pitchers.csv')
pitchers_25 = pd.read_csv('2025_pitchers.csv')
pitchers = pd.concat([pitchers_24,pitchers_25])
del pitchers_24, pitchers_25
pitchers['player_name'] = pitchers['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
old_pitch = pd.read_csv('final_rates_p.csv')
pitch_stats = modify_pitchers(pitchers, old_pitch,players,ids)


del batters, old_hits, old_pitch, pitchers

#%% bullpens

"""
the biggest reason we pulled rosters was for this. I haven't gotten far enough
to be able to acurately predict which relief pitcher will come into the game,
so instead I am pulling every player on the team below a certain BF threshold,
and aggregating their stats so as to get an aggregate bullpen performance
"""

teams = list(pitch_stats.Stadium.unique())
bullpen_stats = pd.DataFrame()
for team in teams:
    scope = pitch_stats.query('Stadium == @team')
    scope = scope.query('avg_bf < 7.0')
    pitch_list = list(scope.pitch.unique())
    for pitch in pitch_list:
        pbp = scope.query('pitch == @pitch')
        pbp = pbp.groupby('pitch').agg(
            pred_hr = ('pred_hr','mean'),
            count = ('count','sum')).reset_index().round(2)
        pbp['Stadium'] = team
        bullpen_stats = bullpen_stats.append(pbp)
    bullpen_stats = bullpen_stats.reset_index(drop=True)
    scope = bullpen_stats.query('Stadium == @team')
    total_count = scope['count'].sum()
    bullpen_stats.loc[bullpen_stats['Stadium'] == team, 'percentage'] = bullpen_stats.loc[bullpen_stats['Stadium'] == team, 'count'] / total_count

bullpen_stats = bullpen_stats.round(3)

del teams, scope, pbp, total_count, pitch_list, pitch, team

#%% get lineup data via rotowire

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
while g < 50:
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

#%% ratings for matchups



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

#%% player odds


"""
pulling player HR odds from popular sportsbooks: in this case just fanduel and
draftkings"""

games = int(len(lineups.query('lineup_spot == 1'))/2)
games = 15
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
    df = df[['Bet Name','FD','DK']]
    all_odds = all_odds.append([df])
    driver.back()
all_odds = all_odds.reset_index(drop=True)
driver.close()
del games, opts, driver_path, driver, event,timer,url,link,tab,tab_html,df,q,i

# fixing df and getting odd diffs

split_df = all_odds['Bet Name'].str.split(' ', expand=True)
try: split_df.columns = ['0', '1', '2','3','4','5']
except ValueError:split_df.columns = ['0', '1', '2','3','4']
split_df[['player','bet','amount']] = 'zero'
for i in range(0,len(all_odds)):
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
split_df[['FanDuel','DraftKings']] = all_odds[['FD','DK']]
try: split_df = split_df.drop(columns=['0','1','2','3','4','5'])
except KeyError: split_df = split_df.drop(columns=['0','1','2','3','4'])
split_df = split_df.query('bet == "Over" and amount == "0.5"')
split_df = split_df[['player','FanDuel','DraftKings']]
split_df = split_df.sort_values(by='FanDuel',ascending=True)
split_df = split_df.drop_duplicates(subset='player',keep='first')
split_df['player'] = split_df['player'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
lineups = lineups.merge(split_df,on='player')
del split_df,all_odds,i

end = datetime.now()
time = end-start
del start,end

"""
after collecting sufficient data, we found that the mean rating was 5, and 
every 1 in 7 batters produced a home run. So to calculate odds, we are claiming that
any hitter with a 5.0 rating has a 14.3% chance to hit a homerun (+800), and any
decimal point either above or below that will act as a modifier to the percentage.

For example, if the matchup rating was 10.0, that player would have a 28.6%, or
+250, chance of hitting a home run, since 5 is double 10"""

lineups['pred_hr'] = round(0.143*(lineups.rating/5),3)*100
lineups['pred_odds'] = round(((100/(lineups.pred_hr))*100)-100)

"""
after calculating our own odds, we will compare them to the more favorable of the
sportsbooks we pulled, and produce the difference in our odds vs theirs, in this
case, a higher number would be seen as more percieved value over the sports books"""

lineups['diff'] =0
for i in range(len(lineups)):
        lineups['diff'][i] = max([lineups.FanDuel[i],lineups.DraftKings[i]])-lineups.pred_odds[i]

"""After that, we want to have a mathmatically consistent method of slecting our
'picks' for the website. 


After looking at the data, we found that a predicted
chance of 18.0% is where the hitters really start to outperform the model, so 
out pick must have at least that high of a percentage

Once it's narrowed down to just those hitters, their difference between predicted
and actual odds must be at least 200. There is no real math behind this selection,
I just didn't want their to be too many selections each day.
"""

good = lineups.query('pred_hr >= 18.0')
good = good.sort_values(by='diff',ascending = False).reset_index(drop=True)
good['pick'] = good.apply(lambda row: 1 if row['diff'] >= 200 else 0, axis = 1)
good = good[['team','player','pick']]
lineups = lineups.merge(good,on=['player','team'],how='left')
lineups['pick'] = lineups['pick'].fillna(0)
lineups.to_csv('daily_lineups.csv',index=False)
#%% this will be used to get results from games the day prior

"""
this section is used strictly to record which hitters hit a home run on the prior day,
once pybaseball updates their game logs."""


from datetime import date, timedelta
import pybaseball as bb
import pandas as pd
lineups=pd.read_csv('daily_lineups.csv')
lineups['HR'] = 0
lineups = lineups.query('rating > 0').reset_index(drop=True)
lineups['date'] = str(date.today()-timedelta(1))
for i in range(0,len(lineups)):
    stats = bb.statcast_batter(lineups.date[i],lineups.date[i], int(lineups.playerid[i]))
    if 'home_run' in list(stats.events):
        lineups.HR[i] = 1
    elif stats.empty:
        lineups = lineups.drop(i)
    else:
        continue
del stats,i

archive = pd.read_csv('pick_archive.csv')
archive = archive.append(lineups)
archive.to_csv('pick_archive.csv',index=False)

"""
archive = pd.read_csv('ratings_archive.csv')
archive = archive.append(lineups)
archive.to_csv('ratings_archive.csv',index=False)
#%% pred ratings
import matplotlib.pyplot as plt
bins = list(range(0, 27)) + [float('inf')]
labels = [f'{i}-{i+1}' for i in range(0, 26)] + ['26+']
archive['bin'] = pd.cut(archive['pred_hr'], bins=bins, labels=labels, right=False, include_lowest=True)

hr_by_bin = archive.groupby('bin')['HR'].mean().rolling(3, min_periods=1).mean()
hr_by_bin.plot(kind='bar', figsize=(12, 6))
plt.title('Home Run Rate by Rating Bin (3-Period Rolling Average)')
plt.xlabel('Rating Bins')
plt.ylabel('HR Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""