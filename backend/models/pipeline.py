import pandas as pd
import numpy as np
from .model import make_preds, get_X_and_y
from .get_game_stats_data import get_game_stats_data_df

def pred_old_outcomes_pipeline(season_year, target_team_ids=None, target_game_date=None, training_and_testing=False):
    df = get_game_stats_data_df(season_year, target_team_ids=target_team_ids, target_game_date=target_game_date, training_and_testing=training_and_testing)
    if df is None:
        return None, None, None, None, None, None
    X, y, home_ids = get_X_and_y(df)
    
    final_preds, final_acc, final_recall, final_precision, final_f1, final_cm = make_preds(X, y, verbose=False)
    if final_preds is None:
        return None, None, None, None, None, None

    outcomes_preds = dict()
    for i in range(len(final_preds)):
        outcomes_preds[home_ids[i]] = (list(y)[i], final_preds[i])

    return outcomes_preds, final_acc, final_recall, final_precision, final_f1, final_cm


import matplotlib.pyplot as plt

def trend_line_graph(plot_data, plot_type, season_year):
    x = range(len(plot_data))
    y = plot_data
    plt.scatter(x, y, label="Data Points")
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color="red", label="Trend Line")
    plt.title(f'{plot_type} Trend Over Time ({season_year} Season)')
    plt.xlabel('Game Days Since First Game')
    plt.ylabel(plot_type)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'ml_imgs/{season_year}_{plot_type.lower()}_trend_line.png')
    plt.show()

def test_model_preds_over_time(season_year):
    df = pd.read_sql_table(f"game_stats_{season_year}", "sqlite:///../database/game_stats.db")
    df = df[df['SEASON_ID'] == f'2{season_year[:season_year.index("-")]}']
    df.sort_values('GAME_DATE', inplace=True)
    game_dates = df['GAME_DATE'].unique()

    accs = []
    recalls = []
    precision = []
    f1 = []
    for game_date in game_dates:
        if len(df[df['GAME_DATE'] == game_date]) < 1:
            print(f"Game date {game_date} is empty")

        outcomes_preds, final_acc, final_recall, final_precision, final_f1, final_cm = pred_old_outcomes_pipeline(
            season_year, target_game_date=game_date, training_and_testing=True
        )

        if outcomes_preds is None:
            print(f"Had no proper data from {game_date}")
        else:
            accs.append(final_acc)
            recalls.append(final_recall)
            precision.append(final_precision)
            f1.append(final_f1)
            print(game_date)

    trend_line_graph(accs, "Accuracy", season_year)
    trend_line_graph(recalls, "Recall", season_year)
    trend_line_graph(precision, "Precision", season_year)
    trend_line_graph(f1, "F1", season_year)
