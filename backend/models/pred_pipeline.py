import pandas as pd
import numpy as np
from .historic_model import get_X_and_y
from .get_game_stats_data import get_game_stats_data_df
import matplotlib.pyplot as plt
from .monte_carlo_simulation import get_all_preds, display_all_reports
import json
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
)


def pred_historic_model_old_outcomes_pipeline(
    league, season_year, user_min_acc, target_team_ids=None,
    target_game_date=None, training_and_testing=False
):
    df = get_game_stats_data_df(
        league,
        season_year,
        target_team_ids=target_team_ids,
        target_game_date=target_game_date,
        training_and_testing=training_and_testing,
    )

    X, y, home_ids, game_dates = get_X_and_y(df)
    det_preds, det_probs, sim_probs, CIs = get_all_preds(
        X, y, league, num_sims=1000
    )

    if det_preds is None:
        return (None,) * 7
    print('hear1')
    with open(
            f"backend/models/acc_thresholds/{league}_acc_per_thresholds_home.json", "r"
    ) as f:
        acc_per_thresholds_home = json.load(f)

    print('here2')
    min_threshold = acc_per_thresholds_home[str(user_min_acc)][1]
    sim_preds = (np.array(sim_probs) > min_threshold).astype(int)

    outcomes_preds = {
        f"{game_date}:{home_id}": (true_label, det_pred, det_prob[0],
                                   sim_pred, sim_prob, CI) for
        home_id, game_date, true_label, det_pred, det_prob,
        sim_pred, sim_prob, CI in zip(
            home_ids, game_dates, y, det_preds, det_probs, sim_preds,
            sim_probs, CIs
        )
    }

    det_acc = accuracy_score(y, det_preds)
    det_recall = recall_score(y, det_preds, zero_division=0)
    det_precision = precision_score(y, det_preds, zero_division=0)
    det_f1 = f1_score(y, det_preds, zero_division=0)
    det_cm = confusion_matrix(y, det_preds, labels=[1, 0])
    sim_acc = accuracy_score(y, sim_preds)
    sim_recall = recall_score(y, sim_preds, zero_division=0)
    sim_precision = precision_score(y, sim_preds, zero_division=0)
    sim_f1 = f1_score(y, sim_preds, zero_division=0)
    sim_cm = confusion_matrix(y, sim_preds, labels=[1, 0])
    accs = (det_acc, sim_acc)
    recalls = (det_recall, sim_recall)
    precisions = (det_precision, sim_precision)
    f1s = (det_f1, sim_f1)
    cms = (det_cm, sim_cm)

    extra_metrics = display_all_reports(y, det_preds, sim_preds)

    return (
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
    )


def trend_line_graph(plot_data, plot_type, season_year):
    x = range(len(plot_data))
    y = plot_data
    plt.scatter(x, y, label="Data Points")
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color="red", label="Trend Line")
    plt.title(f'{plot_type} Trend Over Time ({season_year} Season)')
    plt.xlabel('Game Days Since First Game')
    plt.ylabel(plot_type)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'ml_imgs/{season_year}_{plot_type.lower()}_trend_line.png')
    plt.show()


def eval_model_preds_over_time(league, season_year):
    df = pd.read_sql_table(
        f"{league}_game_stats_{season_year}",
        f"sqlite:///../database/{league}_game_stats.db"
        )
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

        (outcomes_preds, final_acc, final_recall, final_precision,
         final_f1, final_cm) = pred_historic_model_old_outcomes_pipeline(
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


if __name__ == '__main__':
    test_nba_pred_old_outcomes_pipeline = True
    test_ncaa_pred_old_outcomes_pipeline = False
    do_eval_model_over_time = False

    def print_results(
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
    ):
        print("=" * 100)
        if outcomes_preds:
            print(outcomes_preds)
            print(f"Det accuracy: {accs[0]:.4f}")
            print(f"Det recall: {recalls[0]:.4f}")
            print(f"Det precision: {precisions[0]:.4f}")
            print(f"Det F1 Score: {f1s[0]:.4f}")
            print("Det confusion matrix:")
            print(cms[0])
            print(f"Sim accuracy: {accs[1]:.4f}")
            print(f"Sim recall: {recalls[1]:.4f}")
            print(f"Sim precision: {precisions[1]:.4f}")
            print(f"Sim F1 Score: {f1s[1]:.4f}")
            print("Sim confusion matrix:")
            print(cms[1])
            print(extra_metrics)
        else:
            print("No predictions could be made")
        print("=" * 100)

    if test_nba_pred_old_outcomes_pipeline:
        league = "nba"
        season_year = "2024-25"
        user_min_acc = 50
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                league, season_year, user_min_acc,
                target_team_ids=[
                    1610612747, 1610612757, 1610612744, 1610612755,
                    1610612760, 1610612746, 1610612750, 1610612738,
                    1610612749, 1610612751, 1610612748, 1610612754
                ],
                target_game_date="2025-01-02",
                training_and_testing=True
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )

    if test_ncaa_pred_old_outcomes_pipeline:
        league = "ncaa"
        season_year = "2024-25"
        user_min_acc = 60
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                league, season_year, user_min_acc,
                training_and_testing=True
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )

    if do_eval_model_over_time:
        league = "nba"
        season_year = "2024-25"
        eval_model_preds_over_time(league, season_year)
