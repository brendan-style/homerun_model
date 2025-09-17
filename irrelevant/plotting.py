# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 22:20:37 2025

@author: Brendan
"""
#%%

import pandas as pd
archive = pd.read_excel('archives.xlsx')
all_ratings = pd.read_csv('ratings_since_july.csv')
all_ratings['date'] = pd.to_datetime(all_ratings['date'], format='%Y/%m/%d').dt.date
all_ratings['date'] = all_ratings['date'].astype(str)
all_ratings = all_ratings.append(archive).drop_duplicates().reset_index(drop=True)
all_picks = all_ratings.query('pick == 1')
sum(all_ratings.profit)/(len(all_picks)*10)
#%%
import pandas as pd
from itertools import product
import numpy as np
archive = pd.read_excel('archives.xlsx')
archive.pred_odds = (((archive.pred_odds)/(archive.pred_odds-100))*100).round(1)
archive['diff'] = archive.sb_no_hr-archive.pred_odds
dates = list(archive.date.unique())
#dates = dates[8:]
archive = archive.query('date == @dates')
def backtest_thresholds(df, diff_thresholds=None, pred_hr_thresholds=None, bet_amount=10):
    """
    Backtest different threshold combinations for sports betting picks on UNDER bets.
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
        diff_thresholds = [-10000, 0, -1, -1.5, -2, -2.5, -3, -4, -5,-6,-7,-8,-9,-10]
    
    if pred_hr_thresholds is None:
        pred_hr_thresholds = [92, 90, 88, 86, 85, 84, 82, 80, 78, 75, 65, 60, 0]
    
    results = []
    
    # Test each combination of thresholds
    for diff_thresh, pred_hr_thresh in product(diff_thresholds, pred_hr_thresholds):
        
        # Create a copy of the dataframe to work with
        df_copy = df.copy()
        
        # Assign picks based on thresholds (1 if meets criteria, 0 if not)
        # For under bets: we want high under prediction (low HR probability) and positive diff
        df_copy['pick_calculated'] = np.where(
            (df_copy['diff'] <= diff_thresh) & (df_copy['sb_no_hr'] >= pred_hr_thresh), 
            1, 
            0
        )
        
        # Calculate profit for each row based on picks
        df_copy['calculated_profit'] = 0.0
        
        for idx in df_copy.index:
            if df_copy.loc[idx, 'pick_calculated'] == 1:
                # Get the odds
                odds = df_copy.loc[idx, 'odds']
                
                if pd.isna(odds):
                    # Skip this pick if no odds available
                    continue
                
                if df_copy.loc[idx, 'hr'] == 0:
                    # WIN: Player did NOT hit a home run
                    if odds > 0:
                        profit = (odds / 100) * bet_amount
                    else:
                        profit = (100 / abs(odds)) * bet_amount
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
    results_df = results_df.sort_values('roi', ascending=False).reset_index(drop=True)
    
    return results_df
results = backtest_thresholds(archive)
#%% plot
from itertools import product
import numpy as np
import pandas as pd
archive = pd.read_excel('archives.xlsx')
#results = backtest_thresholds(all_ratings)

# roi plot
import matplotlib.pyplot as plt

all_picks = archive.query('pick == 1')
# Assuming your dataframe is called 'df'
# First, ensure date column is datetime
#all_picks['date'] = pd.to_datetime(all_ratings['date'])

# Sort by date to ensure proper time series
df_sorted = all_picks.sort_values('date').copy()

# Calculate cumulative profit over time
df_sorted['cumulative_profit'] = df_sorted['profit'].cumsum()
df_sorted['rolling_roi_10'] = (df_sorted['profit'].rolling(window=10).sum() / 100) * 100
df_sorted['investment'] = 10
df_sorted['cumulative_roi'] = (df_sorted['cumulative_profit'] / df_sorted['investment'].cumsum()) * 100

# Create the plot
plt.figure(figsize=(12, 6))

# For smoothing, we can use a rolling average or interpolation
# Option 1: Simple rolling average (uncomment if desired)
window_size = 3
df_sorted['smoothed_profit'] = df_sorted['cumulative_profit'].rolling(window=window_size, center=True).mean()
plt.plot(df_sorted['date'], df_sorted['smoothed_profit'], linewidth=3, color='steelblue')

# Option 2: Basic line without markers (cleaner look)
#plt.plot(df_sorted['date'], df_sorted['cumulative_profit'], 
 #        linewidth=2.5, color='steelblue')

plt.title('Cumulative Profit Over Time ($10 Wagers)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Profit', fontsize=12)
#plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)

# Format x-axis dates
plt.xticks(plt.xticks()[0][::7])
plt.xticks(rotation=45)
plt.tight_layout()

# Add horizontal line at break-even (y=0)
#plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
# Add ROI text annotation
plt.text(0.25, 0.87, 'ROI: 5.1%', transform=plt.gca().transAxes, 
         fontsize=14, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='square', facecolor='lightblue', alpha=0.8))
#%%
import pandas as pd
from statistics import mean
stats = pd.read_csv('ratings_archive.csv')
picks = pd.read_csv('pick_archive.csv')

picks = picks.drop(columns='pick')
stats['pred_hr'] = round(stats.rating/mean(stats.rating)*(sum(stats.HR/len(stats))),3)*100
stats['pred_odds'] = round(((100/(stats.pred_hr))*100)-100)
stats['diff'] =0
for i in range(len(stats)):
        stats['diff'][i] = max([stats.FanDuel[i],stats.DraftKings[i]])-stats.pred_odds[i]

all_ratings = picks.append(stats).reset_index(drop=True)
all_ratings['sb_odds'] = 0
for i in range(len(all_ratings)):
        all_ratings['sb_odds'][i] = min([all_ratings.FanDuel[i],all_ratings.DraftKings[i]])
all_ratings['sb_hr'] = round(1/((all_ratings.sb_odds+100)/100),3)*100
all_ratings['sb_under_odds'] = round(100*(-1*((100-all_ratings.sb_hr)/100)/((all_ratings.sb_hr)/100)))
all_ratings['sb_under_pred'] = ((100-all_ratings.sb_hr))
all_ratings['under_pred'] = ((100-all_ratings.pred_hr))
all_ratings['under_diff']
        
import matplotlib.pyplot as plt
from statistics import mean
bins = list(range(0, 29)) + [float('inf')]
labels = [f'{i}-{i+1}' for i in range(0, 28)] + ['28+']
all_ratings['bucket'] = pd.cut(all_ratings['pred_hr'], bins=bins, labels=labels, right=False, include_lowest=True)
plt.hist(all_ratings.sb_odds,range=(100,2000),bins=30)

hr_by_bin = all_ratings.groupby('bucket')['HR'].mean().rolling(3, min_periods=1).mean()
hr_by_bin.plot(kind='bar', figsize=(12, 6))
plt.title('Home Run Rate by Rating Bin (3-Period Rolling Average)')
plt.xlabel('Rating Bins')
plt.ylabel('HR Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

all_ratings['sb_odds'] = 0
for i in range(len(all_ratings)):
        all_ratings['sb_odds'][i] = min([all_ratings.FanDuel[i],all_ratings.DraftKings[i]])
all_ratings['sb_hr'] = round(1/((all_ratings.sb_odds+100)/100),3)*100
all_ratings['sb_under_odds'] = round(100*(-1*((100-all_ratings.sb_hr)/100)/((all_ratings.sb_hr)/100)))
all_ratings['sb_under_pred'] = ((100-all_ratings.sb_hr))


plt.figure(figsize=(10, 6))
colors = ['red' if hr == 0 else 'green' for hr in all_ratings['HR']]
plt.scatter(all_ratings['sb_hr'], all_ratings['pred_hr'], c=colors, alpha=0.7)
plt.xlabel('Sportsbook HR%')
plt.ylabel('Your Model HR%')
plt.title('Your Model vs Sportsbooks')
# Add diagonal line
min_val = min(all_ratings['sb_hr'].min(), all_ratings['pred_hr'].min())
max_val = max(all_ratings['sb_hr'].max(), all_ratings['pred_hr'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
plt.legend(['No HR', 'Hit HR'])
plt.show()


#%% Plot 2: Success rate by value (diff) ranges - more granular
diff_ranges = [(-33000, -1000), (-1000, -500), (-500, -200), (-200, -100), (-100, -50), 
               (-50, 0), (0, 50), (50, 100), (100, 200), (200, 500), (500, 1500)]
range_labels = [f"{r[0]} to {r[1]}" for r in diff_ranges]
success_rates = []
mean_pred_hr = []
counts = []

for min_diff, max_diff in diff_ranges:
    mask = (all_ratings['diff'] >= min_diff) & (all_ratings['diff'] < max_diff)
    subset = all_ratings[mask]
    if len(subset) > 0:
        success_rate = subset['HR'].mean() * 100
        mean_hr_pct = subset['pred_hr'].mean()
        count = len(subset)
        success_rates.append(success_rate)
        mean_pred_hr.append(mean_hr_pct)
        counts.append(count)
    else:
        success_rates.append(0)
        mean_pred_hr.append(0)
        counts.append(0)

# Create double bar chart
fig, ax = plt.subplots(figsize=(14, 8))

x = range(len(range_labels))
width = 0.35

# Plot both bars
bars1 = ax.bar([i - width/2 for i in x], success_rates, width, label='Actual HR Rate %', color='tab:blue', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], mean_pred_hr, width, label='Mean Predicted HR %', color='tab:red', alpha=0.7)

ax.set_xlabel('Diff Range (Your Perceived Value)')
ax.set_ylabel('Home Run Percentage')
ax.set_title('Actual vs Predicted HR Rate by Perceived Value\n(Higher diff = better perceived value by your model)')
ax.set_xticks(x)
ax.set_xticklabels(range_labels, rotation=45, ha='right')
ax.legend()

# Add count labels above the bars
for i, count in enumerate(counts):
    if count > 0:
        max_height = max(success_rates[i], mean_pred_hr[i])
        ax.text(i, max_height + 1, f'n={count}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

calude = all_ratings[['sb_odds','pred_odds','HR']]
#%%
pred_odds_ranges = [(-100, -350), (-350, -450), (-450, -550), (-550, -650), (-650, -750), 
                    (-750, -850), (-850, -1000), (-1000, -1200), (-1200, -1500), (-1500, -5000)]
range_labels = [f"{r[0]}-{r[1]}" for r in pred_odds_ranges]

actual_hr_rates = []
your_hr_percentages = []
sb_hr_percentages = []
sample_counts = []

for min_odds, max_odds in pred_odds_ranges:
    mask = (archive['odds'] >= min_odds) & (archive['odds'] < max_odds)
    subset = archive[mask]
    
    if len(subset) > 0:
        # Actual HR rate (what really happened) - percentage of 1's in HR column
        actual_rate = subset['HR'].mean() * 100
        
        # Your mean predicted HR percentage for this odds range
        your_percentage = subset['pred_hr'].mean()
        
        # Sportsbook mean predicted HR percentage for this odds range
        sb_percentage = subset['sb_hr'].mean()
        
        sample_count = len(subset)
        
        actual_hr_rates.append(actual_rate)
        your_hr_percentages.append(your_percentage)
        sb_hr_percentages.append(sb_percentage)
        sample_counts.append(sample_count)
    else:
        actual_hr_rates.append(0)
        your_hr_percentages.append(0)
        sb_hr_percentages.append(0)
        sample_counts.append(0)

# Create triple bar chart
fig, ax = plt.subplots(figsize=(15, 8))
x = range(len(range_labels))
width = 0.25

# Plot three bars with your specified colors
bars1 = ax.bar([i - width for i in x], actual_hr_rates, width, label='Actual HR Rate %', color='green', alpha=0.8)
bars2 = ax.bar([i for i in x], your_hr_percentages, width, label='Model Predicted HR %', color='blue', alpha=0.8)
bars3 = ax.bar([i + width for i in x], sb_hr_percentages, width, label='Sportsbook Predicted HR %', color='red', alpha=0.8)

ax.set_xlabel('Odds Range')
ax.set_ylabel('Home Run Percentage')
ax.set_title('Actual vs Predicted HR Rates by Odds Ranges')
ax.set_xticks(x)
ax.set_xticklabels(range_labels, rotation=45, ha='right')
ax.legend()

# Add sample size labels
for i, count in enumerate(sample_counts):
    if count > 0:
        max_height = max(actual_hr_rates[i], your_hr_percentages[i], sb_hr_percentages[i])
        ax.text(i, max_height + 0.2, f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

#%% Print calibration summary
print("MLB Home Run Model Calibration Summary:")
print("Range\t\tActual%\tYour%\tSB%\tSample")
for i, label in enumerate(range_labels):
    if sample_counts[i] > 0:
        print(f"{label}\t{actual_hr_rates[i]:.1f}\t{your_hr_percentages[i]:.1f}\t{sb_hr_percentages[i]:.1f}\t{sample_counts[i]}")

import pandas as pd

calibration_data = []
sample_sizes = []
pred_abs_diffs = []
sb_abs_diffs = []

for i, label in enumerate(range_labels):
    if sample_counts[i] > 0:
        pred_diff = your_hr_percentages[i] - actual_hr_rates[i]
        sb_diff = sb_hr_percentages[i] - actual_hr_rates[i]
        
        # Store absolute differences and sample sizes for weighted calculation
        pred_abs_diff = abs(pred_diff)
        sb_abs_diff = abs(sb_diff)
        
        calibration_data.append({
            'odds': label,
            'hr_rate': round(actual_hr_rates[i], 2),
            'pred_hr_rate': round(your_hr_percentages[i], 2),
            'sb_pred_hr': round(sb_hr_percentages[i], 2),
            'pred_diff': round(pred_diff, 2),
            'sb_diff': round(sb_diff, 2),
            'sample_size': sample_counts[i]
        })
        
        sample_sizes.append(sample_counts[i])
        pred_abs_diffs.append(pred_abs_diff)
        sb_abs_diffs.append(sb_abs_diff)

calibration_df = pd.DataFrame(calibration_data)

# Calculate weighted average absolute differences (weighted by sample size)
total_samples = sum(sample_sizes)
weighted_pred_mae = sum(abs_diff * size for abs_diff, size in zip(pred_abs_diffs, sample_sizes)) / total_samples
weighted_sb_mae = sum(abs_diff * size for abs_diff, size in zip(sb_abs_diffs, sample_sizes)) / total_samples
calibration_df['value'] = calibration_df.pred_diff - calibration_df.sb_diff
print(calibration_df)
calibration_df[['lower_bound','upper_bound']] = 0
for i in range(len(calibration_df)):
    if i <= 6: 
        calibration_df.upper_bound[i] = int(calibration_df.odds[i][4:])
        calibration_df.lower_bound[i] = int(calibration_df.odds[i][:3])
    else:
        calibration_df.upper_bound[i] = int(calibration_df.odds[i][5:])
        calibration_df.lower_bound[i] = int(calibration_df.odds[i][:4])

all_ratings['pick'] = 0
for i in range(len(all_ratings)):
    if all_ratings['diff'][i] > 0:
        all_ratings.pick[i] = 1
    else:
        continue
    
all_ratings['profit'] = 0
for i in range(len(all_ratings)):
    if all_ratings.pick[i] == 0:
        continue
    elif all_ratings.pick[i] == 1 and all_ratings.HR[i] == 0:
        all_ratings.profit[i] = -10
    else:
        all_ratings.profit[i] = all_ratings.sb_odds[i]/10
        
sum(all_ratings.profit)
all_ratings['value'] =  all_ratings.pred_hr - all_ratings.sb_hr

from itertools import product
import numpy as np
results = backtest_thresholds(all_ratings)
all_ratings['pick'] = 0
for i in range(len(all_ratings)):
    if all_ratings.sb_under_pred[i] >= 88.0 and all_ratings['diff'][i] <= -150:
        all_ratings.pick[i] = 1
    else:
        continue
all_ratings['profit'] = 0
for i in range(len(all_ratings)):
    if all_ratings.pick[i] == 1:
        if all_ratings.HR[i] == 0:
            all_ratings.profit[i] = round(10/(abs((all_ratings.sb_under_odds[i]/100))),2)
        else:
            all_ratings.profit[i] = -10
    else:
        continue
#%%
def backtest_diff_thresholds(df, diff_thresholds=None, bet_amount=10):
    """
    Backtest different diff threshold values for sports betting picks on UNDER bets.
    Only varies diff threshold, uses all qualifying picks regardless of pred_hr value.
    
    Parameters:
    df: DataFrame with columns ['diff', 'sb_under_pred', 'HR', 'sb_under_odds']
    diff_thresholds: list of diff thresholds to test
    bet_amount: amount wagered per bet (default 10)
    
    Returns:
    DataFrame with results for each diff threshold
    """
    
    # Default thresholds if not provided
    if diff_thresholds is None:
        diff_thresholds = [0, -25, -50, -75, -100, -125, -150, -175, -200, -250, -300, -350, -400]
    
    results = []
    
    # Test each diff threshold
    for diff_thresh in diff_thresholds:
        
        # Create a copy of the dataframe to work with
        df_copy = df.copy()
        
        # Assign picks based only on diff threshold (1 if meets criteria, 0 if not)
        df_copy['pick'] = np.where(df_copy['diff'] <= diff_thresh, 1, 0)
        
        # Calculate profit for each row based on picks
        df_copy['calculated_profit'] = 0.0
        
        for idx in df_copy.index:
            if df_copy.loc[idx, 'pick'] == 1:
                # Get the under odds - use sb_under_odds, fallback to calculated from sb_odds
                if 'sb_under_odds' in df_copy.columns and pd.notna(df_copy.loc[idx, 'sb_under_odds']):
                    odds = df_copy.loc[idx, 'sb_under_odds']
                elif 'sb_odds' in df_copy.columns and pd.notna(df_copy.loc[idx, 'sb_odds']):
                    odds = df_copy.loc[idx, 'sb_odds']  # Assuming this is actually the under odds
                else:
                    # Skip this pick if no odds available
                    continue
                
                if df_copy.loc[idx, 'HR'] == 0:
                    # WIN: Player did NOT hit a home run
                    if odds > 0:
                        profit = (odds / 100) * bet_amount
                    else:
                        profit = (100 / abs(odds)) * bet_amount
                    df_copy.loc[idx, 'calculated_profit'] = profit
                else:
                    # LOSS: Player hit a home run
                    df_copy.loc[idx, 'calculated_profit'] = -bet_amount
            else:
                # No pick: no profit or loss
                df_copy.loc[idx, 'calculated_profit'] = 0.0
        
        # Filter to only the picks made
        picks_made = df_copy[df_copy['pick'] == 1].copy()
        
        if len(picks_made) == 0:
            # No qualifying picks
            results.append({
                'diff_threshold': diff_thresh,
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
        wins = (picks_made['HR'] == 0).sum()  # WIN when HR == 0 for under bets
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
    results_df = results_df.sort_values('roi', ascending=False).reset_index(drop=True)
    
    return results_d
# Example usage:
# Assuming your data is in a DataFrame called 'df'
# results = backtest_thresholds(df)

# To view top 20 results by ROI:
# print(results.head(20))

# To filter for strategies with at least 10 picks:
# profitable_strategies = results[results['num_picks'] >= 10]
# print(profitable_strategies.head(10))

# To see the best ROI strategy:
# best_strategy = results.iloc[0]
# print(f"Best strategy: diff >= {best_strategy['diff_threshold']}, pred_hr >= {best_strategy['pred_hr_threshold']}%")
# print(f"ROI: {best_strategy['roi']}%, Win Rate: {best_strategy['win_rate']}%, Picks: {best_strategy['num_picks']}")

# To create a pivot table for visualization:
# roi_heatmap = results.pivot(index='pred_hr_threshold', columns='diff_threshold', values='roi').fillna(0)
# print(roi_heatmap)

#%% products
import pandas as pd
import numpy as np
from itertools import product

def backtest_product_thresholds(df, product_thresholds=None, bet_amount=10):
    """
    Backtest different product threshold combinations for sports betting picks on UNDER bets.
    Bets that players will NOT hit home runs.
    
    Uses product of diff * sb_under_pred as the threshold criterion.
    
    Parameters:
    df: DataFrame with columns ['diff', 'sb_under_pred', 'HR', 'sb_under_odds']
    product_thresholds: list of product thresholds to test
    bet_amount: amount wagered per bet (default 10)
    
    Returns:
    DataFrame with results for each product threshold
    """
    
    # Default thresholds if not provided
    if product_thresholds is None:
        # Include 0 as litmus test, then negative values since we want product <= threshold
        product_thresholds = [0, -5000, -7500, -10000, -12500, -13200, -15000, -17500, -20000, -25000, -30000]
    
    results = []
    
    # Test each product threshold
    for product_thresh in product_thresholds:
        
        # Create a copy of the dataframe to work with
        df_copy = df.copy()
        
        # Calculate the product for each row
        df_copy['threshold_product'] = df_copy['diff'] * df_copy['sb_under_pred']
        
        # Assign picks based on product threshold (1 if meets criteria, 0 if not)
        # We want product <= threshold (since diff is negative for value, product will be negative)
        df_copy['pick'] = np.where(
            df_copy['threshold_product'] <= product_thresh, 
            1, 
            0
        )
        
        # Calculate profit for each row based on picks
        df_copy['calculated_profit'] = 0.0
        
        for idx in df_copy.index:
            if df_copy.loc[idx, 'pick'] == 1:
                # Get the under odds - use sb_under_odds, fallback to calculated from sb_odds
                if 'sb_under_odds' in df_copy.columns and pd.notna(df_copy.loc[idx, 'sb_under_odds']):
                    odds = df_copy.loc[idx, 'sb_under_odds']
                elif 'sb_odds' in df_copy.columns and pd.notna(df_copy.loc[idx, 'sb_odds']):
                    odds = df_copy.loc[idx, 'sb_odds']  # Assuming this is actually the under odds
                else:
                    # Skip this pick if no odds available
                    continue
                
                if df_copy.loc[idx, 'HR'] == 0:
                    # WIN: Player did NOT hit a home run
                    if odds > 0:
                        profit = (odds / 100) * bet_amount
                    else:
                        profit = (100 / abs(odds)) * bet_amount
                    df_copy.loc[idx, 'calculated_profit'] = profit
                else:
                    # LOSS: Player hit a home run
                    df_copy.loc[idx, 'calculated_profit'] = -bet_amount
            else:
                # No pick: no profit or loss
                df_copy.loc[idx, 'calculated_profit'] = 0.0
        
        # Filter to only the picks made
        picks_made = df_copy[df_copy['pick'] == 1].copy()
        
        if len(picks_made) == 0:
            # No qualifying picks
            results.append({
                'product_threshold': product_thresh,
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
        wins = (picks_made['HR'] == 0).sum()  # WIN when HR == 0 for under bets
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
            'product_threshold': product_thresh,
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
    results_df = results_df.sort_values('roi', ascending=False).reset_index(drop=True)
    
    return results_df