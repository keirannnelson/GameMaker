import pandas as pd

def get_max_avg_col(col_list):
    col_avg = []
    col_max = []

    for i in range(len(col_list)):
        if i == 0:
            col_avg.append(pd.NA)
            col_max.append(pd.NA)
        else:
            col_avg.append(sum(col_list[:i]) / i)
            col_max.append(max(col_list[:i])) 

    return col_avg, col_max


# In[17]:


import numpy as np

def get_col_lstsq(col_list):
    col_list = pd.Series(col_list).interpolate(method='linear').bfill().ffill().tolist()
    A = []
    bias = []
    momentum = []

    for i in range(len(col_list)):
        A.append([1, i+1])
        x_lstsq, _, _, _= np.linalg.lstsq(np.array(A), np.array(col_list[:i+1]), rcond=None)
        bias.append(x_lstsq[0])
        momentum.append(x_lstsq[1])

    return bias, momentum


# In[ ]:


def get_game_stats_data_df(season_year, target_team_ids=None, target_game_date=None, training_and_testing=False):
    #TODO Refactor this function if time permits: some of the fcn calls like df.drop() can be consolidated into less calls
    df = pd.read_sql_table(f"game_stats_{season_year}", "sqlite:///backend/database/game_stats.db")
    if training_and_testing:
        df = df[df['SEASON_ID'] == f'2{season_year[:season_year.index("-")]}']
    if target_team_ids:
        df = df[df['TEAM_ID'].isin(target_team_ids)]
    if target_game_date:
        df = df[df['GAME_DATE'] <= target_game_date]
    if df.empty:
        return None


    df['HOME'] = df['MATCHUP'].apply(lambda x: 'vs.' in x if isinstance(x, str) else False).astype(int)
    features = [
        'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
        'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'PF', 'PTS', 'PLUS_MINUS'
    ]
    df = df.dropna()
    i = 0
    
    for team in df['TEAM_ID'].unique():
        team_df = df[df['TEAM_ID'] == team]
        team_sorted = team_df.sort_values('GAME_DATE')

        for col in features:
            col_list = list(team_sorted[col])
            avg_col, max_col = get_max_avg_col(col_list)
            bias_col, mom_col = get_col_lstsq(col_list)

            new_features = {
                f'{col}_BIAS': bias_col,
                f'{col}_MOM': mom_col,
                f'{col}_AVG': avg_col,
                f'{col}_MAX': max_col
            }
            new_features_df = pd.DataFrame(new_features, index=team_sorted.index)
            team_sorted = pd.concat([team_sorted, new_features_df], axis=1)

        if i == 0:
            teams_df = team_sorted.copy()

        else:
            teams_df = pd.concat([teams_df, team_sorted], ignore_index=True)

        i += 1

    if target_game_date:
        teams_df = teams_df[teams_df["GAME_DATE"] == target_game_date]
    teams_df.drop(['FGM', 'FGA','FG3M', 'FG3A', 'FTM','FTA'], axis=1, inplace=True)

    opp_features = [f'{i}_OPP' for i in list(teams_df)]
    home_df = teams_df[teams_df['HOME'] == 1].sort_values('GAME_ID')
    away_df = teams_df[teams_df['HOME'] == 0].sort_values('GAME_ID')
    away_df.columns = opp_features
    common_ids = set(home_df['GAME_ID']) & set(away_df['GAME_ID_OPP'])
    home_df = home_df[home_df['GAME_ID'].isin(common_ids)]
    away_df = away_df[away_df['GAME_ID_OPP'].isin(common_ids)]
    home_df.sort_values('GAME_ID')
    home_df.reset_index(drop=True)
    away_df.sort_values('GAME_ID_OPP')
    away_df.reset_index(drop=True)
    away_df['GAME_ID'] = away_df['GAME_ID_OPP'] 
    merged_df = pd.merge(home_df, away_df, on='GAME_ID')
    merged_df = merged_df.drop([
        'SEASON_ID_OPP',
        'TEAM_ID_OPP',
        'HOME_OPP',
        'MIN_OPP',
        'MATCHUP_OPP',
        'SEASON_ID_OPP',
        'HOME'
    ], axis=1)
    basic_features = [
        'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS'
    ]
    basic_features_opp = [feature + "_OPP" for feature in basic_features][1:]
    merged_df.drop(basic_features + basic_features_opp, inplace=True, axis=1)
    metadata = [
        "SEASON_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "MATCHUP"
    ]
    metadata_opp = ["TEAM_ABBREVIATION_OPP", "TEAM_NAME_OPP", "GAME_ID_OPP"]
    merged_df.drop(metadata + metadata_opp, inplace=True, axis=1)
    merged_df.drop(["REB_BIAS", "REB_MOM", "REB_AVG", "REB_MAX", "REB_BIAS_OPP", "REB_MOM_OPP", "REB_AVG_OPP", "REB_MAX_OPP", "GAME_DATE_OPP", "WL_OPP"], axis=1, inplace=True)
    merged_df.replace({'L': 0, 'W': 1}, inplace=True)
    merged_df.dropna(subset=["TEAM_ID"], inplace=True)
    merged_df.dropna(subset=["MIN_AVG"], inplace=True)
    merged_df.sort_values('GAME_DATE', inplace=True)

    return merged_df