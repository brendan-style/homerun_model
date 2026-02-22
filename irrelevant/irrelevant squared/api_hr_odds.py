# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:51:24 2026

@author: Brendan
"""

"""
pulled odds from sportsgameodds api with free trial 
"""
#%%
from sports_odds_api import SportsGameOdds
import os
import pandas as pd
client = SportsGameOdds(api_key_param=os.environ.get("a50b5ff439a8d749d1a269bbcfa65d21"))
import requests
import json
from datetime import timedelta,datetime

final_results = pd.DataFrame()
for date in pd.date_range('2024-04-01','2025-09-29'):
    date = date.strftime('%Y-%m-%d')
    if date > '2024-09-30' and date < '2025-04-01':
        continue
    end_date = datetime.strptime(date,'%Y-%m-%d') + timedelta(days=1)
    end_date = end_date.strftime('%Y-%m-%d')
    response = requests.get(
        'https://api.sportsgameodds.com/v2/events',
        params={
            'leagueID': 'MLB',
            'startsAfter': f'{date}',
            'startsBefore': f'{end_date}',
            'finalized': 'true',
            'apiKey': 'a50b5ff439a8d749d1a269bbcfa65d21',
            'statID': 'batting_homeRuns',
            'betTypeID': 'ou',
            'statEntityID': 'player',
        }
    )
   
    
    data = response.json()    
    rows = []
    
    for event in data.get('data', []):

        event_date = event.get('status', {}).get('startsAt', 'N/A')
        
        if 'odds' in event:

            hr_odds = {
                odd_id: odd_data 
                for odd_id, odd_data in event['odds'].items() 
                if odd_data.get('statID') == 'batting_homeRuns'
            }
            
            # Process each home run prop
            for odd_id, odd_data in hr_odds.items():
                # Extract player name from the market name
                player_name = odd_data.get('marketName', '').replace(' Home Runs Over/Under', '')
                
                row = {
                    'name': player_name,
                    'date': event_date,
                    'bettypeid': odd_data.get('betTypeID', 'N/A'),
                    'sideid': odd_data.get('sideID', 'N/A'),
                    'bookodds': odd_data.get('bookOdds', 'N/A'),
                    'score': odd_data.get('score', 'N/A'),  # Added score to see results
                    'overunder': odd_data.get('bookOverUnder', 'N/A')  # Added line
                }
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    try: df = df.sort_values(by='name').reset_index(drop=True)
    except KeyError: continue
    final_results = final_results.append(df)
#%%
import numpy as np
# Display the DataFrame
final_results[['new_date','time']] = final_results.date.str.split('T',n=1, expand=True)
final_results = final_results.drop(columns='date').rename(columns={'new_date':'date'})
final_results = final_results.replace('N/A',np.nan).dropna()
final_results = final_results.query('overunder == "0.5"').drop(columns='overunder').reset_index(drop=True)
overs = final_results.query('sideid == "over"').rename(columns={'bookodds':'over'})
unders = final_results.query('sideid == "under"').rename(columns={'bookodds':'under'})
concise_df = overs.merge(unders,on=['name','date','time','score'],how='inner')
concise_df = concise_df.drop(columns=['sideid_x','sideid_y','bettypeid_x','bettypeid_y'])
concise_df['filter'] = concise_df.apply(lambda row: 1 if len(row['name']) <= 1 or len(row['date']) <= 1 else 0, axis = 1)
concise_df = concise_df.query('filter != 1').drop(columns='filter') 
concise_df[['first','last']] = concise_df['name'].str.split(' ',n=1, expand=True)
concise_df = concise_df.drop(columns='name')
concise_df[['over','under']] = concise_df[['over','under']].astype(int)
concise_df.over = ((100/(concise_df.over+100))).round(3)
concise_df.under = ((concise_df.under/(concise_df.under-100))).round(3)
concise_df = concise_df.query('over >= .032 and over <= .465 and under <= 0.99 and under >= .6')
concise_df['total'] = concise_df.over + concise_df.under
concise_df = concise_df.query('total >= 0.979 and total <= 1.11')
concise_df.to_csv('busted_sgo_odds.csv',index=False)
