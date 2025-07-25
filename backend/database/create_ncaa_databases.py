import pandas as pd
from sqlalchemy import create_engine


def csv_to_database(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    engine = create_engine(f'sqlite:///ncaa_game_stats.db')
    df.to_sql(
        csv_file[csv_file.index("/")+1:csv_file.index(".")], engine,
        if_exists='replace', index=False
    )


if __name__ == "__main__":
    csv_to_database("csv/ncaa_game_stats_2024-25.csv")
