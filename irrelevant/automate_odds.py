# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:28:04 2026

@author: Brendan
"""

import time
import pandas as pd
#from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Firefox, FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException #ElementClickInterceptedException
from unidecode import unidecode
from datetime import datetime
#from random import randint
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
    game_time = driver.find_element(By.CSS_SELECTOR,'div.lineup:nth-child('+str(g)+') > div:nth-child(1) > div:nth-child(1)').text
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
                """
                name_check = pitch_stats[pitch_stats['player_name'].str.contains(last_name,case=False)]
                name_check = name_check[name_check['player_name'].str.startswith(initial)]
                if name_check.empty:
                    pass
                else:
                    pitcher['name'][0] = list(name_check['player_name'])[0]
            """
            player = pd.DataFrame([driver.find_element(By.CSS_SELECTOR, 'div.lineup:nth-child('+str(g)+') > div:nth-child(2) > div:nth-child(3) > ul:nth-child('+str(p)+') > li:nth-child('+str(i)+')').text])
            player = player[0].str.split(' ', expand=True)
            player = player.rename(columns={0:'position',(player.shape[1]-1):'handedness'})
            player['name'] = player.iloc[:, 1:-1].apply(lambda x: ' '.join(x), axis=1)
            player = player.drop(columns=player.columns[1:-2])
            if '.' in player['name'][0]:
                last_name = player['name'][0].split()[-1]
                initial = player['name'][0][0]
                """
                name_check = hit_stats[hit_stats['player_name'].str.contains(last_name,case=False)]
                name_check = name_check[name_check['player_name'].str.startswith(initial)]
                if name_check.empty:
                    pass
                else:
                    """
                    #player['name'][0] = list(name_check['player_name'])[0]
                    
                    
            player[['pitcher','p_throws']] = pitcher
            player[['lineup_spot','away_team','home_team','status','first_pitch']] = order,a_team,h_team,status,game_time
            if lineups.empty:
                lineups = player
            else:
                lineups = pd.concat([lineups,player],ignore_index=True)
            order += 1
    g += 1
    lineups = lineups.reset_index(drop=True)    
lineups = lineups.dropna().reset_index(drop=True)

now = datetime.now()
lineups['first_pitch_dt'] = pd.to_datetime(
    lineups['first_pitch'].str.replace(' ET', ''), format='%I:%M %p'
).apply(lambda t: t.replace(year=now.year, month=now.month, day=now.day))
lineups['hours_before'] = (
    (lineups['first_pitch_dt'] - now).dt.total_seconds() / 3600
).round(1).astype(float)
lineups = lineups.drop(columns=['first_pitch','first_pitch_dt'])
lineups['name'] = lineups['name'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
lineups['pitcher'] = lineups['pitcher'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
lineups = lineups.query('hours_before >= 0 and hours_before <= 3').reset_index(drop=True)
lineups['time'] = datetime.today()
driver.quit()
time.sleep(5)

# odds
games = int(len(lineups.query('lineup_spot == 1 and hours_before <= 3'))/2)#and hours_before <= 3
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
driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_WebUserControl_FilterLeague_DropDownListLeague > option:nth-child(4)').click()
timer = driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_WebUserControl_FilterMaximumDateHours_TextBoxMaximumDateHours')
timer.send_keys(12)
driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_ButtonUpdate').click()
all_odds = pd.DataFrame()
for i in range(1,games+1):
    time.sleep(2)
    link = driver.find_element(By.XPATH, '/html/body/form/div[3]/layout-body-main/layout-body-main-right/div[2]/div[2]/table/tbody/tr['+str(i)+']/td[1]/a').text
    """if 'G2' in link:
        continue
    else:"""
    link = driver.find_element(By.XPATH, '/html/body/form/div[3]/layout-body-main/layout-body-main-right/div[2]/div[2]/table/tbody/tr['+str(i)+']/td[1]/a')
    driver.execute_script("arguments[0].removeAttribute('target')", link)
    link.click()
    time.sleep(4)
    for q in range(1,110):
        event = driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListMarket > option:nth-child('+str(q)+')').text
        if event == 'Player Home Runs':
            driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListMarket > option:nth-child('+str(q)+')').click()
            break
        else:
            continue
    time.sleep(4)
    event = driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListSubMarket > option:nth-child(1)').click()
    time.sleep(4)
    tab = driver.find_element(By.TAG_NAME, 'table')
    tab_html = tab.get_attribute('outerHTML')
    df = pd.read_html(tab_html)[0]
    if len(df) < 3:
        driver.refresh()
        time.sleep(2)
        for q in range(1,110):
            event = driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListMarket > option:nth-child('+str(q)+')').text
            if event == 'Player Home Runs':
                driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListMarket > option:nth-child('+str(q)+')').click()
                break
            else:
                continue
        time.sleep(4)
        event = driver.find_element(By.CSS_SELECTOR,'#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListSubMarket > option:nth-child(1)').click()
        time.sleep(4)
        tab = driver.find_element(By.TAG_NAME, 'table')
        tab_html = tab.get_attribute('outerHTML')
        df = pd.read_html(tab_html)[0]
    if i == 1:
        all_odds = df
    else:
        all_odds = pd.concat([all_odds,df],ignore_index=True)
    driver.back()
all_odds = all_odds.reset_index(drop=True)
all_odds['time'] = datetime.today()
driver.close()
#del games, opts, driver_path, driver, event,timer,url,link,tab,tab_html,df,q,i
"""
# fixing df and getting odd diffs
all_odds = all_odds[['Bet Name','MGM','FD','DK','CZR','BR','BB','TSB','FL','CS']].reset_index()
split_df = all_odds['Bet Name'].str.split(' ', expand=True)
try: split_df.columns = ['0', '1', '2','3','4','5']
except ValueError:split_df.columns = ['0', '1', '2','3','4']
split_df[['player','bet','amount']] = 'zero'
for i in range(len(all_odds)):
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
split_df[['MGM','FD','DK','CZR','BR','BB','TSB','FL','CS']] = all_odds[['MGM','FD','DK','CZR','BR','BB','TSB','FL','CS']]
try: split_df = split_df.drop(columns=['0','1','2','3','4','5'])
except KeyError: split_df = split_df.drop(columns=['0','1','2','3','4'])
split_df = split_df.query('amount == "0.5"')
split_df = split_df.fillna(-100000)
split_df = split_df[['player','bet','MGM','FD','DK','CZR','BR','BB','TSB','FL','CS']]
melted = split_df.melt(id_vars=['player', 'bet'], value_vars=['MGM','FD','DK','CZR','BR','BB','TSB','FL','CS'], var_name='book', value_name='odds')
# Separate under rows
under_melted = melted[melted['bet'] == 'Under']
non_cs_under = under_melted[under_melted['book'] != 'CS']

# Best overall odds (max across all books) for Over
max_odds = melted.groupby(['player', 'bet'])['odds'].max().reset_index()
max_books = melted.loc[melted.groupby(['player', 'bet'])['odds'].idxmax()][['player', 'bet', 'book']].rename(columns={'book': 'best_book'})

# CS-specific under odds
cs_under = under_melted[under_melted['book'] == 'CS'][['player', 'odds']].rename(columns={'odds': 'under_circa'})
cs_under['under_circa'] = cs_under['under_circa'].replace(-100000, None)

# Best non-CS under odds
best_non_cs_under_odds = non_cs_under.groupby('player')['odds'].max().reset_index().rename(columns={'odds': 'Under'})
best_non_cs_under_books = non_cs_under.loc[non_cs_under.groupby('player')['odds'].idxmax()][['player', 'book']].rename(columns={'book': 'under_book'})

# Build result from Over only using original logic
result = max_odds[max_odds['bet'] == 'Over'].pivot(index='player', columns='bet', values='odds').reset_index()
book_result = max_books[max_books['bet'] == 'Over'].pivot(index='player', columns='bet', values='best_book').reset_index()
book_result = book_result.rename(columns={'Over': 'over_book'})

result = result.merge(book_result, on='player')

# Merge in non-CS under odds + book
result = result.merge(best_non_cs_under_odds, on='player', how='left')
result = result.merge(best_non_cs_under_books, on='player', how='left')

# Merge in CS under odds
result = result.merge(cs_under, on='player', how='left')

result = result.drop_duplicates(subset='player', keep='first')
result['name'] = result['player'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
result = result.drop(columns='player')
split_df = split_df.fillna(-100000)
split_df = split_df[['player','bet','MGM','FD','DK','CZR','BR','BB','TSB','FL','CS']]
melted = split_df.melt(id_vars=['player', 'bet'], value_vars=['MGM','FD','DK','CZR','BR','BB','TSB','FL','CS'], var_name='book', value_name='odds')
max_odds = melted.groupby(['player', 'bet'])['odds'].max().reset_index()

max_books = melted.loc[melted.groupby(['player', 'bet'])['odds'].idxmax()][['player', 'bet', 'book']].rename(columns={'book': 'best_book'})

result = max_odds.pivot(index='player', columns='bet', values='odds').reset_index()
book_result = max_books.pivot(index='player', columns='bet', values='best_book').reset_index()
book_result = book_result.rename(columns={'Over': 'over_book', 'Under': 'under_book'})
result = result.merge(book_result, on='player')
result = result.drop_duplicates(subset='player',keep='first')
result['name'] = result['player'].apply(unidecode).str.replace(' Jr.', '', regex=True, case=False)
result = result.drop(columns='player')

result = result.fillna(-100000)
"""
ol = pd.read_csv('saving_lineups.csv')
oo = pd.read_csv('saving_odds.csv')
ol = pd.concat([ol,lineups],ignore_index=True)
oo = pd.concat([oo,all_odds],ignore_index=True)
ol.to_csv('saving_lineups.csv',index=False)
oo.to_csv('saving_odds.csv',index=False)
