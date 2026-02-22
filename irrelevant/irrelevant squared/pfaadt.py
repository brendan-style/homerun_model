# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:01:55 2025

@author: Brendan
"""

import pybaseball as bb
import pandas as pd
from numpy import nan
pitchers = pd.read_csv('all_pitchers.csv')
pitchers = pitchers.rename(columns={'last_name, first_name':'name'})
pitchers = pitchers.drop_duplicates(subset=['player_id','name']).reset_index(drop=True)
name_list = ['Pfaadt, Brandon','Detmers, Reid','Alcantara, Sandy','Strider, Spencer','Bradley, Taj'
             ,'Corbin, Patrick','Nola, Aaron','Gibson, Kyle','Glasnow, Tyler','Abbott, Andrew',
             'Assad, Javier','Blanco, Ronel','Snell, Blake','Quintana, Jose','Wacha, Michael','Cole, Gerrit'
             ,'Skenes, Paul','Lugo, Seth','Fedde, Erick']
pitchers = pitchers.query('name == @name_list').reset_index(drop=True)
for i in range(0,len(pitchers)):
    stats = bb.statcast_pitcher("2023-03-01","2025-11-01",pitchers.player_id[i])
    stats = stats.query('game_type == "R"').dropna(subset='pitch_type')
    stats = stats[~stats['pitch_type'].isin(['SC','PO','CS','FA','EP',nan,'AB','FC','IN'])]
    #stats = stats[['player_name','pitcher','game_date','game_year','pitch_type','release_speed','events','description','stand','p_throws',
    #         'bb_type','launch_speed','launch_angle','release_spin_rate','estimated_woba_using_speedangle',
    #         'launch_speed_angle','pfx_x','pfx_z','release_extension','release_pos_x','release_pos_z','inning','outs_when_up','batter']]
    if i == 0:
      pitches_p = stats
    else:
        pitches_p = pitches_p.append(stats)
        
pitches_p = pitches_p[['player_name','p_throws','release_pos_x','release_pos_z','release_speed','pitch_type',
                       'zone','pfx_x','pfx_z','release_spin_rate',
                       'release_extension','launch_angle','launch_speed',
                       'estimated_woba_using_speedangle','woba_value','babip_value',
                       'iso_value','launch_speed_angle','spin_axis','arm_angle',
                       'attack_angle','attack_direction','swing_path_tilt']]

pitch_group = pitches_p.groupby(['pitch_type','p_throws']).mean().reset_index()
counts = pitches_p.groupby(['pitch_type','p_throws']).agg(count=('launch_angle','size')).reset_index()
pitch_group = pitch_group.merge(counts, on=['pitch_type','p_throws'])
