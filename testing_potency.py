# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:39:47 2025

@author: Brendan
"""

#%% testing unders
import pandas as pd
from itertools import product
import numpy as np
def backtest_unders(df, diff_thresholds=None, pred_hr_thresholds=None, bet_amount=10):
    """
    Backtest different threshold combinations for sports betting picks on UNDER bets.
    Bets that players will NOT hit home runs
    
    Parameters:
    df: DataFrame with columns ['diff', 'pred_odds', 'hr', 'odds']
    diff_thresholds: list of diff thresholds to test
    pred_hr_thresholds: list of sb_no_hr thresholds to test (as percentages)
    bet_amount: amount wagered per bet (default 10)
    
    Returns:
    DataFrame with results for each threshold combination
    """
    
    # Default thresholds if not provided
    if diff_thresholds is None:
        diff_thresholds = [.02,.025,.03,.035,.04,.045,.05,.055,.06,.065,.07,.075]
    
    if pred_hr_thresholds is None:
        pred_hr_thresholds = [.92, .90, .88, .86, .85, .84, .82, .80,]
        #pred_hr_thresholds = [.06, .08, .1, .12, .14, .15, .16, 18, .2, .25]
    
    results = []
    
    # Test each combination of thresholds
    for diff_thresh, pred_hr_thresh in product(diff_thresholds, pred_hr_thresholds):
        
        # Create a copy of the dataframe to work with
        df_copy = df.copy()
        
        # Assign picks based on thresholds (1 if meets criteria, 0 if not)
        # For under bets: we want high under prediction (low HR probability) and positive diff
        df_copy['pick_calculated'] = np.where(
            (df_copy['under_diff'] >= diff_thresh) & (df_copy['under'] >= pred_hr_thresh), 
            1, 
            0
        )
        
        # Calculate profit for each row based on picks
        df_copy['calculated_profit'] = 0.0
        
        for idx in df_copy.index:
            if df_copy.loc[idx, 'pick_calculated'] == 1:
                # Get the odds
                odds = df_copy.loc[idx, 'under']
                
                if pd.isna(odds):
                    # Skip this pick if no odds available
                    continue
                
                if df_copy.loc[idx, 'hr'] == 0:
                    # WIN: Player did NOT hit a home run
                    profit = round((10/odds) - 10,2)
                    df_copy.loc[idx, 'calculated_profit'] = profit
                else:
                    # LOSS: Player hit a home run
                    df_copy.loc[idx, 'calculated_profit'] = -bet_amount
            else:
                # No pick: no profit or loss
                df_copy.loc[idx, 'calculated_profit'] = 0.0
        
        # Filter to only the picks made
        picks_made = df_copy[df_copy['pick_calculated'] == 1].copy()
        
        if len(picks_made) == 0:
            # No qualifying picks
            results.append({
                'diff_threshold': diff_thresh,
                'pred_hr_threshold': pred_hr_thresh,
                'num_picks': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_wagered': 0,
                'roi': 0,
                'avg_profit_per_pick': 0,
                'profit_per_day': 0
            })
            continue
        
        # Calculate metrics
        num_picks = len(picks_made)
        wins = (picks_made['hr'] == 0).sum()  # WIN when hr == 0 for under bets
        losses = num_picks - wins
        win_rate = (wins / num_picks) * 100 if num_picks > 0 else 0
        
        # Calculate profit using our calculated profits
        total_profit = picks_made['calculated_profit'].sum()
        total_wagered = num_picks * bet_amount
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        avg_profit_per_pick = total_profit / num_picks if num_picks > 0 else 0
        
        # Calculate profit per day
        unique_dates = picks_made['date'].nunique() if 'date' in picks_made.columns else 1
        profit_per_day = total_profit / unique_dates if unique_dates > 0 else 0
        
        results.append({
            'diff_threshold': diff_thresh,
            'pred_hr_threshold': pred_hr_thresh,
            'num_picks': num_picks,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'total_profit': round(total_profit, 2),
            'total_wagered': total_wagered,
            'roi': round(roi, 2),
            'avg_profit_per_pick': round(avg_profit_per_pick, 2),
            'profit_per_day': round(profit_per_day, 2)
        })
    
    # Convert to DataFrame and sort by ROI descending
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_profit', ascending=False).reset_index(drop=True)
    
    return results_df

#%% testing overs
import pandas as pd
from itertools import product
import numpy as np
def backtest_overs(df, diff_thresholds=None, pred_hr_thresholds=None, bet_amount=10):
    """
    Backtest different threshold combinations for sports betting picks on OVER bets.
    Bets that players will NOT hit home runs.
    
    Parameters:
    df: DataFrame with columns ['diff', 'pred_odds', 'hr', 'odds']
    diff_thresholds: list of diff thresholds to test
    pred_hr_thresholds: list of sb_no_hr thresholds to test (as percentages)
    bet_amount: amount wagered per bet (default 10)
    
    Returns:
    DataFrame with results for each threshold combination
    """
    
    # Default thresholds if not provided
    if diff_thresholds is None:
        diff_thresholds = [.02,.025,.03,.035,.04,.045,.05,.055,.06,.065,.07,.075]
    
    if pred_hr_thresholds is None:
        pred_hr_thresholds = [.1,.12,.14, .15, .16, .18, .2, .22, .25]
        #pred_hr_thresholds = [.06, .08, .1, .12, .14, .15, .16, 18, .2, .25]
    
    results = []
    
    # Test each combination of thresholds
    for diff_thresh, pred_hr_thresh in product(diff_thresholds, pred_hr_thresholds):
        
        # Create a copy of the dataframe to work with
        df_copy = df.copy()
        
        # Assign picks based on thresholds (1 if meets criteria, 0 if not)
        # For under bets: we want high under prediction (low HR probability) and positive diff
        df_copy['pick_calculated'] = np.where(
            (df_copy['over_diff'] >= diff_thresh) & (df_copy['over'] >= pred_hr_thresh), 
            1, 
            0
        )
        
        # Calculate profit for each row based on picks
        df_copy['calculated_profit'] = 0.0
        
        for idx in df_copy.index:
            if df_copy.loc[idx, 'pick_calculated'] == 1:
                # Get the odds
                odds = df_copy.loc[idx, 'over']
                
                if pd.isna(odds):
                    # Skip this pick if no odds available
                    continue
                
                if df_copy.loc[idx, 'hr'] == 1:
                    # WIN: Player hit a home run
                    profit = round((10/odds) - 10,2)
                    df_copy.loc[idx, 'calculated_profit'] = profit
                else:
                    # LOSS: Player hit a home run
                    df_copy.loc[idx, 'calculated_profit'] = -bet_amount
            else:
                # No pick: no profit or loss
                df_copy.loc[idx, 'calculated_profit'] = 0.0
        
        # Filter to only the picks made
        picks_made = df_copy[df_copy['pick_calculated'] == 1].copy()
        
        if len(picks_made) == 0:
            # No qualifying picks
            results.append({
                'diff_threshold': diff_thresh,
                'pred_hr_threshold': pred_hr_thresh,
                'num_picks': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_wagered': 0,
                'roi': 0,
                'avg_profit_per_pick': 0,
                'profit_per_day': 0
            })
            continue
        
        # Calculate metrics
        num_picks = len(picks_made)
        wins = (picks_made['hr'] == 1).sum()  # WIN when hr == 0 for under bets
        losses = num_picks - wins
        win_rate = (wins / num_picks) * 100 if num_picks > 0 else 0
        
        # Calculate profit using our calculated profits
        total_profit = picks_made['calculated_profit'].sum()
        total_wagered = num_picks * bet_amount
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        avg_profit_per_pick = total_profit / num_picks if num_picks > 0 else 0
        
        # Calculate profit per day
        unique_dates = picks_made['date'].nunique() if 'date' in picks_made.columns else 1
        profit_per_day = total_profit / unique_dates if unique_dates > 0 else 0
        
        results.append({
            'diff_threshold': diff_thresh,
            'pred_hr_threshold': pred_hr_thresh,
            'num_picks': num_picks,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'total_profit': round(total_profit, 2),
            'total_wagered': total_wagered,
            'roi': round(roi, 2),
            'avg_profit_per_pick': round(avg_profit_per_pick, 2),
            'profit_per_day': round(profit_per_day, 2)
        })
    
    # Convert to DataFrame and sort by ROI descending
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_profit', ascending=False).reset_index(drop=True)
    
    return results_df


#%%

