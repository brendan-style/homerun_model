# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 12:38:56 2025

@author: Brendan


GET STARTING PITCHER MATCHUPS WHEN DOING STATS PULL
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
username = 'brendanstylewt@gmail.com'
password = '8fnpa33X$Vvv=9W'
#driver.close()
#options = Options()
opts = FirefoxOptions()
opts.add_argument("--width=950")
opts.add_argument("--height=1025")
driver_path = "C:\\Users\\brend\\Downloads\\geckodriver-v0.33.0-win-aarch64(1).zip\\geckodriver.exe"
driver = Firefox(options=opts)
#driver = webdriver.Chrome(chrome_options = options, executable_path = driver_path)
url = "https://therundown.io/"
driver.get(url)
time.sleep(3)

# Signing In
driver.find_element(By.CSS_SELECTOR,'button.Button_buttonBase__ha3oR:nth-child(1) > span:nth-child(1)').click()
driver.find_element(By.CSS_SELECTOR,'a.Button_fullWidth__xiCcj:nth-child(1)').click()
driver.find_element(By.CSS_SELECTOR,'#id_username').send_keys(username)
driver.find_element(By.CSS_SELECTOR,'#id_password').send_keys(password)
driver.find_element(By.CSS_SELECTOR,'.btn').click()
time.sleep(2)

# Navigating to MLB Section
driver.find_element(By.CSS_SELECTOR,'button.Button_buttonBase__ha3oR:nth-child(1) > span:nth-child(1)').click()
driver.find_element(By.CSS_SELECTOR,'.BurgerMenu_burgerMenuContent__3I2dk > ul:nth-child(2) > nav:nth-child(1) > li:nth-child(1) > button:nth-child(1) > div:nth-child(1)').click()
driver.find_element(By.CSS_SELECTOR,'.BurgerGroup_expanded__wbUaS > div:nth-child(3) > button:nth-child(1) > div:nth-child(1) > div:nth-child(1)').click()
driver.find_element(By.CSS_SELECTOR,'.BurgerGroup_expanded__wbUaS > div:nth-child(3) > section:nth-child(2) > li:nth-child(2) > div:nth-child(1) > a:nth-child(1)').click()
time.sleep(1)

# adjusting date

driver.find_element(By.CSS_SELECTOR,'button.Button_tertiaryLarge__OT0qm:nth-child(1)').click()
monthyear = driver.find_element(By.CSS_SELECTOR,'.Heading_baseHeading__TfdmH').text
while monthyear != 'April 2023':
    driver.find_element(By.CSS_SELECTOR,'.CustomDatePicker_customHeader__lNiuR > button:nth-child(1) > span:nth-child(1) > span:nth-child(1)').click()
    monthyear = driver.find_element(By.CSS_SELECTOR,'.Heading_baseHeading__TfdmH').text
driver.find_element(By.CSS_SELECTOR,'.react-datepicker__day--001').click()
time.sleep(1)
# Going game by game - then advancing to next date
# have to search team by team since games switch formatting after 10 matchups
all_odds = pd.DataFrame()
#%% code failed often on my laptop, so sectioned this off for easy rerun



while True:
    #get date for day, establish list of teams to search
    date = pd.Series(driver.find_element(By.CSS_SELECTOR,'button.Button_tertiaryLarge__OT0qm:nth-child(1)').text)
    if date[0][0:2] == '10': break
    else: pass
    mlb_teams = ['Orioles','White Sox','Astros','Braves','Cubs','Diamondbacks','Red Sox','Guardians','Angels','Marlins','Reds','Rockies','Yankees','Tigers','Athletics','Mets','Brewers','Dodgers','Rays','Royals','Mariners','Phillies','Cardinals','Padres','Blue Jays','Twins','Rangers','Nationals','Pirates','Giants']
    day_odds = pd.DataFrame()
    
    while len(mlb_teams) != 0:
        # search team
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR,'.TableHeader_searchInput__W7zbH').send_keys(mlb_teams[0])
        time.sleep(1)
        # Pull teams and remove them both from list
        try:
            away = pd.Series(driver.find_element(By.CSS_SELECTOR,'div.TeamsOnly_row__UfsQ6:nth-child(1) > div:nth-child(1) > a:nth-child(1) > div:nth-child(1) > div:nth-child(2) > span:nth-child(1) > span:nth-child(2)').text)
        except NoSuchElementException:
            mlb_teams.pop(0)
            time.sleep(1)
            driver.find_element(By.CSS_SELECTOR,'.TableHeader_searchWrapper__2jWNV > div:nth-child(1) > button:nth-child(3) > svg:nth-child(1)').click()
            continue
        home = pd.Series(driver.find_element(By.CSS_SELECTOR,'div.TeamsOnly_row__UfsQ6:nth-child(2) > div:nth-child(1) > a:nth-child(1) > div:nth-child(1) > div:nth-child(2) > span:nth-child(1) > span:nth-child(2)').text)
        mlb_teams.remove(home[0])
        mlb_teams.remove(away[0])
        # navigate to game
        driver.find_element(By.CSS_SELECTOR,'.EventMeta_viewGame__pqzM7').click()
        time.sleep(1)
        # click on prop bets
        driver.find_element(By.XPATH, "//li[contains(@class, 'game-primary-tab') and contains(text(), 'Props')]").click()                        
        driver.find_element(By.CSS_SELECTOR,'button.Button_buttonBase__ha3oR:nth-child(3)').click()
        time.sleep(1)
        # if no filters, game has no bets, so code exits
        filters = driver.find_element(By.CSS_SELECTOR,'.TableFilter_filterCountText__EBhh9').text
        if filters == '0 / 0': 
            driver.find_element(By.CSS_SELECTOR,'.GameDetailHeader_backArrow__M1xjW').click()
            time.sleep(1)
            continue
        else: pass
        # edit prop bets so HR odds are only odds listed
        driver.find_element(By.CSS_SELECTOR,'div.TableFilter_filterGroup__LORdR:nth-child(3) > button:nth-child(2)').click()
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR,'.CustomModal_modalFooter__3rHNG > div:nth-child(1) > div:nth-child(1) > button:nth-child(2)').click()
        # for 2024, always first option. 2025, placement differs
        # 2024 code
        #driver.find_element(By.CSS_SELECTOR,'label.Checkbox_label__O-SB1:nth-child(1) > div:nth-child(2)').click()
        # 2025 code
        for t in range(1,int(filters[0:2])):
            try: event = driver.find_element(By.CSS_SELECTOR,'label.Checkbox_label__O-SB1:nth-child('+str(t)+')')
            except NoSuchElementException: continue
            if 'home run' in event.text:
                event.click()
                break
            else:
                continue
        driver.find_element(By.CSS_SELECTOR,'.CustomModal_modalFooter__3rHNG > div:nth-child(1) > button:nth-child(2)').click()
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR,'button.FilterModal_tab__pRst\+:nth-child(2)').click()
        driver.find_element(By.CSS_SELECTOR,'.CustomModal_modalFooter__3rHNG > div:nth-child(1) > div:nth-child(1) > button:nth-child(1)').click()
        driver.find_element(By.CSS_SELECTOR,'.CustomModal_modalFooter__3rHNG > div:nth-child(1) > button:nth-child(2)').click()
        # pull odds
        ct = 2
        row_ct = 1
        # not formatted as table so row indexing is required
        # Rows are formatted in pairs so indexing numbers are weird
        while True:
            try:
                prop = pd.Series(driver.find_element(By.CSS_SELECTOR,'div.PropTable_linesRow__kJAOy:nth-child('+str(ct)+') > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > span:nth-child('+str(row_ct)+')').text)
                odds = pd.Series(driver.find_element(By.CSS_SELECTOR,'div.PropTable_linesRow__kJAOy:nth-child('+str(ct)+') > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(5) > div:nth-child(1) > span:nth-child('+str(row_ct)+')').text)
                data = pd.concat([prop,odds,date,away,home],axis=1,keys=['prop', 'odds','date','away','home'])
                day_odds = day_odds.append(data)
                if row_ct == 1:
                    row_ct += 1
                else:
                    row_ct -= 1
                    ct += 1
            except NoSuchElementException: break
        driver.find_element(By.CSS_SELECTOR,'.GameDetailHeader_backArrow__M1xjW').click()
        time.sleep(1)
    day_odds = day_odds.reset_index(drop=True)
    all_odds = all_odds.append(day_odds)
    driver.find_element(By.CSS_SELECTOR,'button.Button_buttonBase__ha3oR:nth-child(3) > span:nth-child(1)').click()
    time.sleep(1)
    
all_odds = all_odds.drop_duplicates()
all_odds = all_odds.reset_index(drop=True)
all_odds.to_csv('historical_odds_multi.csv',index=False)

#%% editing odds df
import pandas as pd
all_odds = pd.read_csv('historical_odds_24.csv')
split_df = all_odds['prop'].str.split('\n', expand=True)
split_df[['2','3']] = split_df[1].str.split(' ',expand=True)
split_df = split_df.drop(columns=1)
split_df[['name','prop','amount']] = split_df[[0,'2','3']]
all_odds = all_odds.drop(columns='prop')
all_odds[['name','prop','amount']] = split_df[['name','prop','amount']]
all_odds = all_odds.query('odds != "PK"')
all_odds['odds'] = all_odds['odds'].str.replace('+', '')
all_odds['amount'] = all_odds['amount'].str.replace('0,5', '0.5')
all_odds['odds'] = all_odds['odds'].astype(int)
all_odds = all_odds.query('amount == "0.5"')
all_odds = all_odds.drop(columns='amount')
all_odds = all_odds[['name','prop','odds','away','home','date']]

#issue where some odds were incorrect
test = all_odds.query('prop == "Over" and odds <= 100')
other_test = all_odds.query('prop == "Under" and odds >= -100')
all_odds = all_odds.drop(test.index.tolist() + other_test.index.tolist()).reset_index(drop=True)
all_odds.to_csv('historical_odds_multi.csv',index=False)
