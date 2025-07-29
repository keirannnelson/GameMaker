import pandas as pd
import numpy as np
import warnings


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


def get_col_lstsq(col_list):
    col_list = pd.Series(col_list).interpolate(
        method='linear'
        ).bfill().ffill().tolist()
    A = []
    bias = []
    momentum = []

    for i in range(len(col_list)):
        A.append([1, i + 1])
        x_lstsq, _, _, _ = np.linalg.lstsq(
            np.array(A), np.array(col_list[:i + 1]), rcond=None
            )
        bias.append(x_lstsq[0])
        momentum.append(x_lstsq[1])

    return bias, momentum


def get_new_row_data(base_data, num_times_seen, n):
    for i in range(len(base_data)):
        if i % 4 == 2:
            y_new = base_data.iloc[i-1] * num_times_seen + base_data.iloc[i-2]
            base_data.iloc[i] = ((n + num_times_seen) * base_data.iloc[i] +
                                y_new)/(n + num_times_seen + 1)
        elif i % 4 == 3:
            base_data.iloc[i] = max(base_data.iloc[i-1], base_data.iloc[i])

    return list(base_data)


def get_game_stats_data_df(
    league, season_year, target_team_ids=None, target_game_dates=None,
    target_game_ids=None, training_and_testing=False
):
    target_team_ids = list(set(target_team_ids))
    df = pd.read_sql_table(
        f"{league}_game_stats_{season_year}",
        f"sqlite:///backend/database/{league}_game_stats.db"
    )
    if training_and_testing and league == "nba":
        season_prefix = f"2{season_year.split('-')[0]}"
        df = df[df['SEASON_ID'] == season_prefix]
    if target_team_ids:
        df = df[df['TEAM_ID'].isin(target_team_ids)]
    target_game_dates.sort()
    if target_game_dates:
        df = df[df['GAME_DATE'] <= target_game_dates[-1]]
    df['HOME'] = df['MATCHUP'].apply(
        lambda x: int('vs.' in x) if isinstance(x, str) else 0
    )

    n_for_each_team = {team_id: 0 for team_id in target_team_ids}
    for index, row in df.iterrows():
        if row['GAME_DATE'] < target_game_dates[0]:
            team_id = row['TEAM_ID']
            n_for_each_team[team_id] = n_for_each_team[team_id] + 1

    if "LEAGUE" in df.columns:
        df.drop(
            columns=["LEAGUE"], inplace=True, errors='ignore'
        )
    
    features = [
        'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
        'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'PF', 'PTS', 'PLUS_MINUS'
    ]
    df = df.dropna(axis=0, how='any')
    team_dfs = []
    for team in df['TEAM_ID'].unique():
        team_df = df[df['TEAM_ID'] == team].sort_values('GAME_DATE').copy()

        for col in features:
            col_list = team_df[col].tolist()
            avg_col, max_col = get_max_avg_col(col_list)
            bias_col, mom_col = get_col_lstsq(col_list)

            new_features = pd.DataFrame({
                f'{col}_BIAS': bias_col,
                f'{col}_MOM': mom_col,
                f'{col}_AVG': avg_col,
                f'{col}_MAX': max_col
            }, index=team_df.index)

            team_df = pd.concat([team_df, new_features], axis=1)

        team_dfs.append(team_df)

    teams_df = pd.concat(team_dfs, ignore_index=True)
    if target_game_dates:
        teams_df = teams_df[teams_df["GAME_DATE"].isin(target_game_dates)]
    teams_df.drop(
        ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA'], axis=1,
        inplace=True
    )
    opp_features = [f"{col}_OPP" for col in teams_df.columns]
    home_df = teams_df[teams_df['HOME'] == 1].sort_values('GAME_ID').copy()
    away_df = teams_df[teams_df['HOME'] == 0].sort_values('GAME_ID').copy()
    away_df.columns = opp_features
    common_game_ids = set(home_df['GAME_ID']).intersection(
        set(away_df['GAME_ID_OPP'])
    )
    home_df = home_df[home_df['GAME_ID'].isin(common_game_ids)].reset_index(
        drop=True
    )
    away_df = away_df[away_df['GAME_ID_OPP'].isin(common_game_ids)].reset_index(
        drop=True
    )
    away_df['GAME_ID'] = away_df['GAME_ID_OPP']
    merged_df = pd.merge(home_df, away_df, on='GAME_ID')
    columns_to_drop = [
        'SEASON_ID_OPP', 'HOME_OPP', 'MIN_OPP', 'MATCHUP_OPP',
        'HOME',
    ]
    merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    basic_features = [
        'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
        'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS'
    ]
    basic_features_opp = [f"{feat}_OPP" for feat in basic_features][1:]
    merged_df.drop(
        columns=basic_features + basic_features_opp, inplace=True,
        errors='ignore'
    )
    metadata = [
        "SEASON_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "MATCHUP"
    ]
    metadata_opp = ["TEAM_ABBREVIATION_OPP", "TEAM_NAME_OPP", "GAME_ID_OPP"]
    merged_df.drop(
        columns=metadata + metadata_opp, inplace=True, errors='ignore'
    )
    reb_cols = [
        "REB_BIAS", "REB_MOM", "REB_AVG", "REB_MAX",
        "REB_BIAS_OPP", "REB_MOM_OPP", "REB_AVG_OPP", "REB_MAX_OPP",
        "GAME_DATE_OPP", "WL_OPP"
    ]
    merged_df.drop(columns=reb_cols, inplace=True, errors='ignore')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        merged_df.replace({'L': 0, 'W': 1}, inplace=True)
    merged_df.dropna(subset=["TEAM_ID", "MIN_AVG"], inplace=True)
    merged_df.sort_values('GAME_DATE', inplace=True)
    merged_df = merged_df[merged_df["GAME_ID"].isin(target_game_ids)]
    merged_df.drop(columns=['GAME_ID'], inplace=True)

    if len(target_game_dates) > 1:
        team_ids = list(
            set(
                list(merged_df["TEAM_ID"]) + list(merged_df["TEAM_ID_OPP"])
            )
        )
        team_ids_seen = {team_id: 0 for team_id in team_ids}
        team_ids_base_data = {team_id: [] for team_id in team_ids}


        merged_df_iter = merged_df.iterrows()
        for index, row in merged_df_iter:
            team_id = row['TEAM_ID']
            team_id_opp = row['TEAM_ID_OPP']

            if team_ids_seen[team_id] == 0:
                team_ids_base_data[team_id] = row.loc["MIN_BIAS":"PLUS_MINUS_MAX"]
            else:
                merged_df.loc[index, "MIN_BIAS":"PLUS_MINUS_MAX"] = get_new_row_data(
                    team_ids_base_data[team_id], team_ids_seen[team_id],
                    n_for_each_team[team_id]
                )

            if team_ids_seen[team_id_opp] == 0:
                team_ids_base_data[team_id_opp] = row.loc["MIN_BIAS_OPP":"PLUS_MINUS_MAX_OPP"]
            else:
                merged_df.loc[index, "MIN_BIAS_OPP":"PLUS_MINUS_MAX_OPP"] = get_new_row_data(
                    team_ids_base_data[team_id_opp], team_ids_seen[team_id_opp],
                    n_for_each_team[team_id_opp]
                )

            team_ids_seen[team_id] = team_ids_seen[team_id] + 1
            team_ids_seen[team_id_opp] = team_ids_seen[team_id_opp] + 1

    merged_df.drop(columns=["TEAM_ID_OPP"], inplace=True, errors='ignore')

    return merged_df


if __name__ == "__main__":
    # test_df1 = get_game_stats_data_df(
    #     "nba",
    #     "2023-24",
    #     target_team_ids=[
    #         1610612742, 1610612760, 1610612753, 1610612749, 1610612757,
    #         1610612758
    #     ],
    #     target_game_dates="2024-04-14",
    #     training_and_testing=True
    #
    # )
    # print(test_df1)

    test_df1 = get_game_stats_data_df(
        "nba",
        "2024-25",
        target_team_ids=[1610612760, 1610612737, 1610612751],
        target_game_dates=["2024-10-17", "2024-10-23"],
        target_game_ids=['0012400064', '0022400064'],
        training_and_testing=False

    )
    print(test_df1)

    # test_df2 = get_game_stats_data_df(
    #     "nba",
    #     "2023-24",
    #     target_team_ids=None,
    #     target_game_dates=None,
    #     training_and_testing=True
    # )
    # print(test_df2)
