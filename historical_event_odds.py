""" Sample code to query historical odds for any market for a single event

Historical odds are only available on paid usage plans.

More information can be found at https://the-odds-api.com/historical-odds-data/
"""

import argparse
import json

import requests


# Obtain the api key that was passed in from the command line
parser = argparse.ArgumentParser(description='Historical odds sample code')
parser.add_argument('--api-key', type=str, default='')
args = parser.parse_args()


# An api key is emailed to you when you sign up to a plan
# Get a free API key at https://api.the-odds-api.com/
api_key = args.api_key or 'ea6c75b333fe56984afae6903574e208'

# Sport key
# More info at https://the-odds-api.com/sports-odds-data/sports-apis.html
sport = 'baseball_mlb'

# Bookmaker regions
# uk | us | eu | au. Multiple can be specified if comma delimited.
# More info at https://the-odds-api.com/sports-odds-data/bookmaker-apis.html
regions = 'us' 

# Odds markets
# Multiple can be specified if comma delimited
# More info at https://the-odds-api.com/sports-odds-data/betting-markets.html
markets = 'batter_home_runs' 

# Odds format
# decimal | american
odds_format = 'american'

# Date format
# iso | unix
date_format = 'iso'

# Historical timestamp
# Must be in ISO8601 format
date = '2025-09-01T17:11:00Z'

# Event ID
# A list of event ids at this timestamp can be found using the events endpoint, for example:
# https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events?apiKey={API_KEY}&date={DATE}
# This event id is for the Trail Blazers @ Pistons game on 2023-11-01
event_id = '7964df1255e389e562a90cf9fb91244b'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# Query bookmaker odds for live and upcoming games as they were at the specified DATE parameter.
# The usage quota cost = 10 x [number of markets specified] x [number of regions specified]
# For examples of usage quota costs, see https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs-3
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

odds_response = requests.get(f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{event_id}/odds', params={
    'api_key': api_key,
    'regions': regions,
    'markets': markets,
    'oddsFormat': odds_format,
    'dateFormat': date_format,
    'date': date,
})

if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

else:
    odds_json = odds_response.json()

    print(json.dumps(odds_json['data'], indent=4))

    print(f"Timestamp: {odds_json['timestamp']}")
    print(f"Previous available timestamp: {odds_json['previous_timestamp']}")
    print(f"Next available timestamp: {odds_json['next_timestamp']}")
    
    # Check the usage quota
    print('Remaining credits', odds_response.headers['x-requests-remaining'])
    print('Used credits', odds_response.headers['x-requests-used'])
#%%

import argparse
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Historical odds sample code')
parser.add_argument('--api-key', type=str, default='')
args = parser.parse_args()
api_key = args.api_key or 'ea6c75b333fe56984afae6903574e208'
sport = 'baseball_mlb'
regions = 'us'
markets = 'batter_home_runs'
odds_format = 'american'
date_format = 'iso'

#season_odds = pd.DataFrame(columns=['player_name', 'over_under', 'odds', 'point','sportsbook', 'date'])

start_date = datetime(2025, 8, 19)
end_date = datetime(2025, 9, 28)
season_odds = pd.read_csv('historical_odds_2025.csv')
#%%
while start_date < end_date:
    date = start_date.strftime('%Y-%m-%dT00:00:00Z')
    response = requests.get(f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events', params={
        'api_key': api_key,
        'date': date
    })
    
    if response.status_code == 200:
        event_data = response.json().get('data', [])
        for event in event_data:
            event_id = event['id']
            if pd.to_datetime(event['commence_time']).strftime('%Y-%m-%d') != start_date.strftime('%Y-%m-%d'):
                continue
            odds_response = requests.get(f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{event_id}/odds', params={
                'api_key': api_key,
                'regions': regions,
                'markets': markets,
                'oddsFormat': odds_format,
                'dateFormat': date_format,
                'date': event['commence_time']
            })
            
            if odds_response.status_code == 200:
                odds_json = odds_response.json()
                all_sbs = odds_json['data']['bookmakers']
                for i in range(len(odds_json['data']['bookmakers'])):
                    data = odds_json['data']['bookmakers'][i]['markets'][0]['outcomes']
                    for q in range(len(data)):
                        season_odds = season_odds.append({
                            'player_name': data[q]['description'],
                            'over_under': data[q]['name'],
                            'odds': data[q]['price'],
                            'point': data[q]['point'],
                            'sportsbook': all_sbs[i]['key'],
                            'date': start_date.strftime('%Y-%m-%d')
                        }, ignore_index=True)
    print('Used credits', odds_response.headers['x-requests-used'])
    print('date:', start_date.strftime('%Y-%m-%d'))
    start_date += timedelta(days=1)

season_odds.to_csv('historical_odds_2025.csv',index=False)
#%%
import pandas as pd
odds = pd.read_csv('historical_odds_2025.csv')
odds = odds.query('point == 0.5')
sbs = list(odds.sportsbook.unique())
odds = odds[odds.sportsbook.isin(['fanduel','draftkings','fanatics','betmgm'])]
unders = odds.query('over_under == "Under"')
overs = odds.query('over_under == "Over"')
overs = overs.sort_values(by='odds',ascending=False)
overs = overs.drop_duplicates(subset=['player_name','date','point'],keep='first')
overs = overs.drop(columns=['sportsbook','over_under']).rename(columns={'odds':'over'})
unders = unders.drop(columns=['sportsbook','over_under']).rename(columns={'odds':'under'})
combo = unders.merge(overs,on=['player_name','point','date'],how='left').reset_index(drop=True)
combo = combo[['date','player_name','over','under']]
combo.over = ((100/(combo.over+100))).round(3)
combo.under = ((combo.under/(combo.under-100))).round(3)
combo.to_csv('workable_25_odds.csv',index=False)
#%%
import pandas as pd
from unidecode import unidecode
import pybaseball as bb
matchups = pd.read_csv('workable_25_odds.csv')
bat_names = pd.read_csv('batters.csv')[['player_name','playerid']].drop_duplicates().reset_index(drop=True)
bat_names[['last','first']] = bat_names['player_name'].str.split(', ',expand = True)
bat_names['player_name'] = bat_names['first']+' '+bat_names['last']
bat_names = bat_names.drop(columns=['first','last']).rename(columns={'playerid':'batterid'})
bat_names['player_name'] = bat_names['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
matchups['player_name'] = matchups['player_name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
matchups = matchups.merge(bat_names,how='left',on='player_name').dropna()
matchups.to_csv('workable_25_odds.csv',index=False)
#%%
import pandas as pd
import pybaseball as bb
from pandas.errors import ParserError
import time
matchups = pd.read_csv('workable_25_odds.csv')
matchups[['team','pitcher','hr','pa','stand','p_throws','home','away']] = 0

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
matchups.to_csv('workable_25_odds.csv',index=False)
#%%
import pandas as pd
matchups25 = pd.read_csv('workable_25_odds.csv')