# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:37:19 2026

@author: Brendan
"""


"""
Written by Claude
"""
import requests
from datetime import date, timedelta

def get_hourly_forecast(zip_code, days=7):
    geo = requests.get(f"https://api.zippopotam.us/us/{zip_code}").json()
    lat = geo["places"][0]["latitude"]
    lon = geo["places"][0]["longitude"]
    city = geo["places"][0]["place name"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "windspeed_10m", "weathercode",'relativehumidity_2m'],
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "forecast_days": days
    }
    weather = requests.get(url, params=params).json()
    return city, weather["hourly"]

def get_hour(data, date_str, hour):
    target = f"{date_str}T{hour:02d}:00"
    idx = data["time"].index(target)
    return {key: data[key][idx] for key in data if key != "time"}, target

# Get tomorrow's date
tomorrow = (date.today() + timedelta(days=1)).isoformat()

city, data = get_hourly_forecast("10451")
result, ts = get_hour(data, tomorrow, 19)  # 19 = 7pm

print(f"Forecast for {city} on {ts}:")
print(f"  Temp:   {result['temperature_2m']}°F")
print(f"  Precip: {result['precipitation']}mm")
print(f"  Wind:   {result['windspeed_10m']}mph")
print(f"  Code:   {result['weathercode']} (WMO weather code)")

result['temperature_2m']
result['relativehumidity_2m']
#%% getting day or night for every game in sample
""" ran once to get zip codes for individual games
ids = pd.read_excel('HR_factors.xlsx',dtype={"zip": str,'year':str})[['Stadium','team','zip','year']].drop_duplicates().reset_index(drop=True)
ids = ids.rename(columns={'team':'home'})
archive = pd.read_csv('v2_ratings.csv')
archive['year'] = archive['date'].str[:4]
archive_id = archive.merge(ids,on=['home','year'],how='left').drop_duplicates()
"""

import pybaseball as bb
import pandas as pd
import re
from datetime import datetime
def parse_game_date(date_str, year):
    cleaned = re.sub(r'\(\d\)', '', date_str)          # remove (1), (2)
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', cleaned)
    cleaned = cleaned.split(', ', 1)[-1].strip()
    return datetime.strptime(f"{cleaned} {year}", "%b %d %Y").strftime("%Y-%m-%d")

archive = pd.read_csv('v2_ratings.csv')[['date','Stadium','zip']].drop_duplicates()
archive['year'] = archive['date'].str[:4].astype(int)
team_list=list(archive.Stadium.unique())
year_list=list(archive['date'].str[:4].unique().astype(int))
for year in year_list:
    for team in team_list:
        if team == 'AZ':
            all_games = bb.schedule_and_record(year,'ARI')
        elif team == 'ATH' and year < 2025:
            all_games = bb.schedule_and_record(year,'OAK')
        elif team == 'CWS':
            all_games = bb.schedule_and_record(year,'CHW')
        elif team == 'WSH':
            all_games = bb.schedule_and_record(year,'WSN')
        else:
            all_games = bb.schedule_and_record(year,team)
        all_games['Year'] = year
        all_games['date_clean'] = all_games.apply(lambda row: parse_game_date(row['Date'], row['Year']), axis=1)
        all_games = all_games.query('Home_Away == "Home"').reset_index(drop=True)
        all_games = all_games[['Tm','D/N','date_clean']].rename(columns={'date_clean':'date','Tm':'Stadium'})
        all_games['Stadium'] = team
        subset = archive.query('Stadium == @team and year == @year')
        subset = subset.merge(all_games,on=['Stadium','date'])
        if team == team_list[0] and year == year_list[0]:
            game_times = subset
        else:
            game_times = pd.concat([game_times,subset],ignore_index=True)
game_times.to_csv('gametimes.csv',index=False)
#%% using day/night infor to approximate temperature/humidity for any game
"""
since I haven't found an easy way to get exact game start times using pybaseball,
I will be using the day/night info to approxmiate temperature during the game.

Day - avg of 12pm-4pm
Night - avg of 7pm-10pm

"""
def get_historical_avg(zip_code, date_str, start_hour, end_hour):
    geo = requests.get(f"https://api.zippopotam.us/us/{zip_code}").json()
    lat = geo["places"][0]["latitude"]
    lon = geo["places"][0]["longitude"]
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relativehumidity_2m"],
        "temperature_unit": "fahrenheit",):
        "timezone": "auto",
        "start_date": date_str,
        "end_date": date_str
    }
    data = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params).json()["hourly"]
    
    rows = [
        {key: data[key][i] for key in data if key != "time"}
        for i, ts in enumerate(data["time"])
        if ts.startswith(date_str) and start_hour <= int(ts[11:13]) <= end_hour
    ]
    return {key: round(sum(r[key] for r in rows) / len(rows), 1) for key in rows[0]}

import pandas as pd
game_times = pd.read_csv('gametimes.csv')
game_times = game_times.query('zip >0').reset_index(drop=True) # removing toronto
domes = ['AZ','HOU','MIA','MIL','SEA','TB','TEX','TOR']
for i in range(len(game_times)
    zc = game_times['zip'][i]
    if zc == 2215:
        zc = '02215'
    if game_times['D/N'][i] == 'D':
        avgs = get_historical_avg(zc, date, 12, 16)
    else:
        avgs = get_historical_avg(zc, date, 19, 22)
    avgs = pd.DataFrame([avgs]).rename(columns={'temperature_2m':'temp','relativehumidity_2m':'rel_hum'})
    if i == 0:
        totals = avgs
    else:
        totals = pd.concat([totals,avgs],ignore_index=True)

game_times[['temp','rel_hum']] = totals
game_times.to_csv('gametimes.csv',index=False)
#%%
import pandas as pd
from statistics import mean
domes = ['AZ','HOU','MIA','MIL','SEA','TB','TEX','TOR']
times = pd.read_csv('gametimes.csv')
times = times[~times['Stadium'].isin(domes)]
times = times[['date','temp','rel_hum','Stadium']].rename(columns={'Stadium':'home_team','date':'game_date'})
pitches = pd.read_csv('pitches_p.csv')
fbb = pitches.query('in_play==1').dropna(subset=['estimated_woba_using_speedangle'])
fbb = fbb.merge(times,on=['game_date','home_team'])
fbb = fbb.query('fly_ball == 1')
sum(fbb.home_run)/len(fbb)
# fb/hr = 15%
mean(fbb.temp)
# avg temp: 72
16.3/15
fbb[['launch_speed','launch_angle']] = round(fbb[['launch_speed','launch_angle']])
subset = fbb.groupby('launch_speed')[['estimated_woba_using_speedangle','hit_distance_sc']].mean().reset_index()
subset = subset.sort_values(by='launch_speed',ascending=False).reset_index(drop=True)
subset['change'] = round(subset['estimated_woba_using_speedangle'].pct_change()*100,1)
fbs = fbb.query('launch_speed >= 95 and launch_speed <= 100 and launch_angle >= 25 and launch_angle <= 30').dropna()
fbs[['temp','rel_hum']] = round(fbs[['temp','rel_hum']],-1)
temp = fbs.groupby('rel_hum').agg(
    distance=('hit_distance_sc','mean'),
    count=('rel_hum','size')).reset_index()
#import matplotlib.pyplot as plt
#plt.hist(fbb.launch_speed,bins=30)
#%% removing old weather and park factors
import pandas as pd
times = pd.read_csv('gametimes.csv')[['date','Stadium','temp']]
domes = ['AZ','HOU','MIA','MIL','SEA','TB','TEX','TOR']
archive = pd.read_csv('v2_ratings.csv')
archive = archive.merge(times,on=['date','Stadium']).reset_index(drop=True)
hr_by_month = {3:0.93,4:0.93,5:0.95,6:1.02,7:1.05,8:1.04,9:1.01}
factors = pd.read_excel('HR_factors.xlsx')
for i in range(len(archive)):
    month = archive.month[i]
    bats = archive.stand[i]
    throws = archive.p_throws[i]
    if bats == "S" and throws == 'R':
        bats = 'L'
    elif bats == "S" and throws == 'L':
        bats = 'R'
    stadium = archive.team[i]
    c_year = archive.year[i]
    # remove old weather rating
    archive.rating[i] = round(archive.rating[i]/hr_by_month[month],2)
    # remove old park factors
    park = factors.query('Handedness == @bats and team == @stadium and year == @c_year').reset_index(drop=True)
    archive.rating[i] = round(archive.rating[i]/float(park.HR.values),2)
#%% new weather and stadium 3.98
archive.temp = archive.temp.round()
archive = archive.drop(80228).reset_index()
for i in range(len(archive)):
    #temp
    if archive.Stadium[i] in domes and archive.temp[i] > 80:
        archive.temp[i] = 80
    temp_diff = int(72-archive.temp[i])
    weather_mod = ((15-((temp_diff/10)))/15)
    archive.rating[i] = round(archive.rating[i]*weather_mod,2)
    # park
    bats = archive.stand[i]
    throws = archive.p_throws[i]
    if bats == "S" and throws == 'R':
        bats = 'L'
    elif bats == "S" and throws == 'L':
        bats = 'R'
    park = factors.query('Handedness == @bats and team == @stadium and year == @c_year').reset_index(drop=True)
    archive.rating[i] = round(archive.rating[i]*float(park.savant_hr.values),2)
archive.to_csv('v3_ratings.csv',index=False)
#%%
""" avg hr didn't provide large enough sample, new plan
hrs = fbb.query('home_run == 1')
avgs = hrs.groupby('description')[['launch_speed','launch_angle','hit_distance_sc','temp','rel_hum']].mean()
# looking for balls hit between 104-106 and 27-29 degree launch angle

subset = fbb.query('launch_speed >= 104 and launch_speed <= 106 and launch_angle >= 27 and launch_angle <= 29')
subset[['temp','rel_hum']] = round(subset[['temp','rel_hum']],-1)
temp = subset.groupby('temp')['hit_distance_sc'].mean().reset_index()
"""