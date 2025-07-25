from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import players
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect
import sys
import time


custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
}


def _save_df_as_table(df, table_name, db_engine, if_exists='replace'):
    df.rename(
        columns={col_name: col_name.upper() for col_name in df.columns},
        inplace=True
    )
    df.to_sql(
        table_name, con=db_engine, if_exists=if_exists, index=False
    )


def _update_table(
    db_path, engine, batch_limit, data_list, records_processed, saving_func,
    record_ids=None, sleep_time=0.25
):
    num_data = len(data_list)
    if len(records_processed) != len(data_list):
        if record_ids is None:
            record_ids = data_list

        batch_num = 0
        print(f"\nRetrieving and saving the following data for {db_path}:")
        for i, data in enumerate(data_list):
            if batch_num == batch_limit:
                print(
                    f"Reached your batch limit of {batch_limit} records. "
                    f"Restarting the script..."
                    )
                # Restart script, adhering to session volume limit
                os.execv(sys.executable, [sys.executable] + sys.argv)

            record_id = record_ids[i]
            if str(record_id) in records_processed:
                continue

            try:
                # Slow down the request rate, adhering to rate limit
                time.sleep(sleep_time)
                saving_func(engine, record_id)
                batch_num += 1
                print(
                    f"{i + 1}/{num_data} ({batch_num}/{batch_limit}"
                    f"): Saved record {data} to table at {db_path}"
                )
            except Exception as e:
                print(f"{i + 1}. Error: {e}")
                break

    print(f"All {num_data} tables for {db_path} are saved")


def get_and_save_player_career_stats(
    db_path="player_career_stats.db", batch_limit=-1,
):
    def saving_func_for_player_career_stats(engine, record_id):
        df = playercareerstats.PlayerCareerStats(
            record_id, headers=custom_headers, timeout=30
        ).get_data_frames()[0]
        df.drop('PLAYER_ID', axis=1, inplace=True)
        _save_df_as_table(df, str(record_id), engine, if_exists='append')

    all_players = players.get_players()
    engine = create_engine(f'sqlite:///{db_path}')
    processed_ids = set(inspect(engine).get_table_names())
    record_ids = [player["id"] for player in all_players]
    _update_table(
        db_path, engine, batch_limit, all_players, processed_ids,
        saving_func_for_player_career_stats, record_ids=record_ids
    )


def get_and_save_player_stats_per_game(
    db_path="player_stats_per_game.db", year="2024-25", batch_limit=-1
):

    def saving_func_for_player_stats_per_game(engine, record_id):
        df = boxscoretraditionalv2.BoxScoreTraditionalV2(
            game_id=record_id, headers=custom_headers, timeout=120
        ).player_stats.get_data_frame()
        _save_df_as_table(df, str(record_id), engine, if_exists='append')

    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=year)
    game_ids = gamefinder.get_data_frames()[0]['GAME_ID'].unique().tolist()
    db_path = db_path[:db_path.index(".")] + year + db_path[db_path.index("."):]

    engine = create_engine(f'sqlite:///{db_path}')
    processed_ids = set(inspect(engine).get_table_names())
    _update_table(
        db_path, engine, batch_limit, game_ids, processed_ids,
        saving_func_for_player_stats_per_game, sleep_time=0.3
        )


def get_and_save_player_info(db_path="player_info.db"):
    engine = create_engine(f'sqlite:///{db_path}')
    df = pd.DataFrame(players.get_players())
    df.drop('full_name', axis=1, inplace=True)
    _save_df_as_table(df, "player_info", engine)
    print(f"The table for {db_path} is saved")


def get_and_save_game_stats(db_path="nba_game_stats.db", year="2024-25"):
    engine = create_engine(f'sqlite:///{db_path}')
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=year)
    df = gamefinder.get_data_frames()[0]
    _save_df_as_table(df, f"game_stats_{year}", engine)
    print(f"The table for {db_path} is saved")


def main():
    # Put any and all function calls to the get_and_save-type functions that
    # will restart the script when it hits its batch_limit first; all other
    # functions go last
    # get_and_save_player_career_stats(batch_limit=500)
    years_to_do = ["2022-23", "2021-22", "2020-21"]
    get_and_save_player_stats_per_game(batch_limit=500, year=years_to_do[0])
    # get_and_save_player_info()
    get_and_save_game_stats(year=years_to_do[0])


if __name__ == "__main__":
    main()
