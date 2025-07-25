import json
import pandas as pd
from sqlalchemy import create_engine


def _process_data_vector(row, start_idx):
    data_vector = list(row)[start_idx:]
    min_col_idx = 0

    if data_vector[min_col_idx]:
        data_vector[min_col_idx] = float(
            data_vector[min_col_idx][:data_vector[min_col_idx].index(":")]
        )
        if data_vector[min_col_idx] == 0.0:
            data_vector = []
    else:
        data_vector = []

    return data_vector


def _get_data_for_player(data, team_id, game_id, player_id, row, start_idx):
    if team_id not in data.keys():
        data[team_id] = {}
    if player_id not in data[team_id].keys():
        data[team_id][player_id] = {}
    #if game_id not in data[team_id][player_id]:
    #   data[team_id][player_id][game_id] = []

    data[team_id][player_id][game_id] = _process_data_vector(row, start_idx)


def _get_data_for_game(data, team_id, game_id, player_stats_per_game_conn):
    df = pd.read_sql_query(
        f'''SELECT * FROM "{game_id}" WHERE TEAM_ID = "{team_id}"''', player_stats_per_game_conn
    )
    start_idx = list(df.columns).index("MIN")
    player_ids = df["PLAYER_ID"]

    for i, row in df.iterrows():
        _get_data_for_player(
            data, team_id, game_id, str(player_ids[i]), row, start_idx
        )


def _get_data_for_team(
    player_stats_per_game_engine, data, team_id, game_stats_conn
):
    df = pd.read_sql_query(
        f'''SELECT * FROM game_stats WHERE TEAM_ID = 
        "{team_id}" ORDER BY GAME_DATE DESC''', game_stats_conn
    )
    game_id_col_name = "GAME_ID"
    game_ids = list(df[game_id_col_name])

    with player_stats_per_game_engine.connect() as player_stats_per_game_conn:
        for game_id in game_ids:
            _get_data_for_game(
                data, team_id, str(game_id), player_stats_per_game_conn
            )


def convert_db_to_ml_ready_json(
    game_stats_db_path="sqlite:///../database/nba_game_stats.db",
    player_stats_per_game_db_path="sqlite:///../database/player_stats_per_game.db"
):
    game_stats_engine = create_engine(game_stats_db_path)
    player_stats_per_game_engine = create_engine(player_stats_per_game_db_path)
    with game_stats_engine.connect() as game_stats_conn:
        team_id_col_name = "TEAM_ID"
        df = pd.read_sql_query(
            f'''SELECT DISTINCT {team_id_col_name} FROM game_stats''',
            game_stats_conn
        )
        team_ids = df[team_id_col_name].tolist()
        data = {}
        for team_id in team_ids:
            _get_data_for_team(
                player_stats_per_game_engine, data, str(team_id),
                game_stats_conn
            )

    with open('nba_data.json', 'w') as f:
        json.dump(data, f, indent=4)


def main():
    convert_db_to_ml_ready_json()

    # JSON ends up in the following format:
    # {
    #   TEAM_ID: {
    #           PLAYER_ID: {
    #                   GAME_ID: [
    #                           A list of 20 (float) basketball stats from
    #                           player PLAYER_ID on team TEAM_ID in game
    #                           GAME_ID; an empty list means they played for
    #                           less than a minute or not at all during the
    #                           game. GAME_ID are listed in reverse
    #                           chronological order. Stat labels in order:
    #                           MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM,
    #                           FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK, TO,
    #                           PF, PTS, PLUS_MINUS
    #                            ]
    #                      }
    #           }
    # }


if __name__ == '__main__':
    main()
