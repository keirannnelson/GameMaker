import numpy as np
import pandas as pd
import joblib
import os
from .historic_model import make_preds, get_X_and_y
from sklearn.metrics import classification_report
from .get_game_stats_data import get_game_stats_data_df
import json


def smooth_clip(avg, max_val, alpha=2):
    smooth = 0.95 * max_val * np.tanh(alpha * avg / max_val) / np.tanh(alpha)
    return np.minimum(avg, smooth)


def CI95_percentage(x, n):
    lower_bound = np.round(x - 1.96 * np.sqrt(x * (100 - x) / n), 2)
    upper_bound = np.round(x + 1.96 * np.sqrt(x * (100 - x) / n), 2)
    return [lower_bound, upper_bound]


def get_sim_probs(X, row_idx, folds, num_sims=1000):
    X_row = X.iloc[[row_idx]]
    ensemble_preds = []

    for i in range(len(folds)):
        fold = folds[i]
        X_fold = X_row.reindex(
            columns=fold['feature_names_pre_drop'], fill_value=0
            )
        X_fold.fillna(fold['mean'], inplace=True)

        X_sim = np.tile(X_fold.values, (num_sims, 1))
        X_sim = pd.DataFrame(X_sim, columns=X_fold.columns)

        std_devs = fold['std_dev']
        np.random.seed(42)
        for feature, std_dev in std_devs.items():
            noise = np.random.normal(
                loc=0, scale=0.5*std_dev, size=num_sims
                )
            X_sim[feature] += noise

            if "MAX" in feature and "PCT" not in feature:
                X_sim[feature] = np.round(X_sim[feature])
            elif "PCT" in feature:
                X_sim[feature] = np.clip(X_sim[feature], 0, 1)
            elif "AVG" in feature and "PLUS_MINUS" not in feature:
                X_sim[feature] = np.clip(X_sim[feature], 0, None)

        avg_max_pairs = [
            (feature, feature.replace("AVG", "MAX"))
            for feature in X_sim.columns if "AVG" in feature
        ]
        for avg_feature, max_feature in avg_max_pairs:
            if max_feature in X_sim.columns:
                X_sim[avg_feature] = smooth_clip(
                    X_sim[avg_feature], X_sim[max_feature]
                )

        X_sim.fillna(fold['mean'], inplace=True)
        X_sim = pd.DataFrame(
            fold['scaler'].transform(X_sim),
            columns=X_sim.columns,
            index=X_sim.index
        )
        X_sim.drop(columns=fold['dropped_features'], inplace=True)

        probs = fold['model'].predict_proba(X_sim)[:, 1]
        y_preds = (probs >= fold['best_threshold']).astype(int)
        ensemble_preds.append(y_preds)

    preds_stack = np.vstack(ensemble_preds)
    final_preds = preds_stack.mean(axis=0)
    sim_probs = np.round(100 * final_preds.sum() / num_sims, 2)

    return sim_probs


def get_all_preds(X, y, league, num_sims=1000):
    num_test_cases = X.shape[0]
    X_test = X.iloc[:num_test_cases]

    det_preds = [None] * num_test_cases
    det_probs = [None] * num_test_cases
    sim_probs = [None] * num_test_cases
    CIs = [None] * num_test_cases

    folder_path = f'backend/models/{league}_model_fold_data'
    fold_files = [f for f in os.listdir(folder_path) if
                  os.path.isfile(os.path.join(folder_path, f))]
    num_folds = len(fold_files)
    folds = [joblib.load(f'{folder_path}/ensemble_fold_{i}.pkl') for i in
             range(num_folds)]
    for i in range(num_test_cases):
        print(f"Getting predictions for test case {i+1}/{num_test_cases} "
              f"({np.round(100 * (i+1)/num_test_cases, 2)}%)...")
        x_i = X.iloc[[i]]
        y_i = y.iloc[i]
        det_pred, det_prob, *_ = make_preds(
            x_i, np.array([[y_i]]), league,
            verbose=False,
            get_conf_matrix_img=False
        )
        sim_prob = get_sim_probs(
            X_test, i, folds, num_sims=num_sims
        )

        det_preds[i] = det_pred[0]
        det_probs[i] = det_prob
        sim_probs[i] = sim_prob
        CIs[i] = CI95_percentage(sim_prob, num_sims)

    return det_preds, det_probs, sim_probs, CIs


def display_all_reports(y_test, det_preds, sim_preds):
    def display_report(report, title, display_extras=None):
        info_dict = {}
        separator_char_width = 55
        bullet_point = " - "
        primary_sep = "="
        secondary_sep = "-"
        print()
        print(primary_sep * separator_char_width)
        print(title)
        print(primary_sep * separator_char_width)
        if display_extras:
            for i, extra in enumerate(display_extras):
                print(bullet_point + extra)
                info_dict[f"extra{i+1}"] = extra
        acc = np.round(100 * report['accuracy'], 2)
        print(bullet_point + f"Accuracy:  {acc}%")
        info_dict["accuracy"] = acc
        class_labels = [key for key in report.keys() if key != 'accuracy']
        for class_label in class_labels:
            info_dict[class_label] = {}
            print(secondary_sep * separator_char_width)
            print(
                bullet_point +
                f"Class {class_label} "
                f"({int(report[class_label]['support'])} tot):"
                )
            recall = np.round(report[class_label]['recall'], 4)
            precision = np.round(report[class_label]['precision'], 4)
            f1 = np.round(report[class_label]['f1-score'], 4)
            print(f"\tRecall:    {recall}")
            print(f"\tPrecision: {precision}")
            print(f"\tF1-score:  {f1}")
            info_dict[class_label]["recall"] = recall
            info_dict[class_label]["precision"] = precision
            info_dict[class_label]["f1"] = f1
        print(primary_sep * separator_char_width)

        return info_dict

    info_dict = {}

    y_test = pd.Series(y_test)
    det_right_sim_wrong = [
        1 if d != s and d == t else 0 for d, s, t in zip(
            det_preds, sim_preds, y_test
        )
    ]
    det_target_names = list(set(list(y_test) + list(det_preds)))
    det_report_extras = [
        f"Deterministic-right-simulation-wrong number: "
        f"{sum(det_right_sim_wrong)}"]
    det_report = classification_report(
        y_test, det_preds, target_names=det_target_names, output_dict=True
    )
    title = "deterministic"
    info_dict[title] = display_report(
        det_report, title,
        display_extras=det_report_extras
    )

    sim_right_det_wrong = [
        1 if d != s and s == t else 0 for d, s, t in zip(
            det_preds, sim_preds, y_test
        )
    ]
    sim_target_names = list(set(list(y_test) + list(sim_preds)))
    sim_report_extras = [
        f"Simulation-right-deterministic-wrong number: "
        f"{sum(sim_right_det_wrong)}"]
    sim_report = classification_report(
        y_test, sim_preds, target_names=sim_target_names, output_dict=True
    )
    title = "simulation"
    info_dict[title] = display_report(
        sim_report, title, display_extras=sim_report_extras
    )

    det_sim_agree = [
        1 if d == s else 0 for d, s, t in zip(det_preds, sim_preds, y_test)
    ]
    if len(det_sim_agree) > 0:
        y_test_agree = [y for y, a in zip(y_test, det_sim_agree) if a == 1]
        agree_preds = [p for p, a in zip(det_preds, det_sim_agree) if a == 1]
        agree_target_names = list(set(list(y_test_agree) + list(agree_preds)))
        agree_report_extras = [
            f"Deterministic-and-simulation-agree number: {len(agree_preds)} "
            f"({(100 * len(agree_preds) / len(det_preds)):.2f}%)"
        ]
        agree_report = classification_report(
            y_test_agree, agree_preds, target_names=agree_target_names,
            output_dict=True
        )
        title = "agree"
        info_dict[title] = display_report(
            agree_report, title,
            display_extras=agree_report_extras
        )

    det_sim_disagree = [
        i for i, (d, s) in enumerate(zip(det_preds, sim_preds)) if d != s
    ]
    if len(det_sim_disagree) > 0:
        y_test_disagree = [y_test.iloc[i] for i in det_sim_disagree]
        disagree_report_extras = [
            f"Disagree number: {len(det_sim_disagree)} "
            f"({(100 * len(det_sim_disagree) / len(det_preds)):.2f}%)"
        ]

        det_disagree_preds = [det_preds[i] for i in det_sim_disagree]
        det_target_names = list(
            set(list(y_test_disagree) + list(det_disagree_preds))
        )
        det_disagree_report = classification_report(
            y_test_disagree, det_disagree_preds, target_names=det_target_names,
            output_dict=True
        )
        title = "deterministic-disagree"
        info_dict[title] = display_report(
            det_disagree_report, title,
            display_extras=disagree_report_extras
        )

        sim_disagree_preds = [sim_preds[i] for i in det_sim_disagree]
        sim_target_names = list(
            set(list(y_test_disagree) + list(sim_disagree_preds))
        )
        sim_disagree_report = classification_report(
            y_test_disagree, sim_disagree_preds, target_names=sim_target_names,
            output_dict=True
        )
        title = "simulation-disagree"
        info_dict[title] = display_report(
            sim_disagree_report, title,
            display_extras=disagree_report_extras
        )

    return info_dict


def get_acc_per_thresholds(for_home_team, sim_probs, y, verbose=True):
    def get_nearest_acc(acc):
        return int((acc // 10) * 10)

    default_acc_per_thresholds_tuple = (0, 101, 0, 0)
    acc_per_thresholds = {
        10 * i: default_acc_per_thresholds_tuple for i in range (11)
    }
    thresholds = np.arange(0, 100, 0.01)
    sim_probs = np.array(sim_probs)
    y = np.array(y)

    if not for_home_team:
        sim_probs = 100 - sim_probs

    for threshold in thresholds:
        threshold = np.round(threshold, 2)
        sub_y = y[sim_probs >= threshold]

        if len(sub_y) > 0:
            accuracy = 0 if len(sub_y) == 0 else np.round(
                100 * np.count_nonzero(
                    sub_y == int(for_home_team))/len(sub_y), 2
            )
            nearest_acc = get_nearest_acc(accuracy)
            per_games = np.round((100 * len(sub_y)/len(sim_probs)), 2)

            if for_home_team:
                if (accuracy >= nearest_acc and 
                        threshold < acc_per_thresholds[nearest_acc][1]):
                    acc_per_thresholds[nearest_acc] = (
                        accuracy, threshold, per_games, len(sub_y)
                    )
            else:
                if (accuracy >= nearest_acc and 
                        threshold < acc_per_thresholds[nearest_acc][1]):
                    acc_per_thresholds[nearest_acc] = (
                        accuracy, threshold, per_games, len(sub_y)
                    )

            if verbose:
                print(f"Minimum simulation confidence threshold: {threshold}")
                print(
                    f"Number of games that meet simulation threshold: "
                    f"{len(sub_y)} ({per_games}%)"
                )
                print(f"Accuracy for those games:  {accuracy}%")
                print("="*100)

    key_value_pairs = list(acc_per_thresholds.items())
    for key, value in key_value_pairs:
        if value == default_acc_per_thresholds_tuple or key < 50:
            del acc_per_thresholds[key]

    return acc_per_thresholds


if __name__ == "__main__":
    league = "ncaa"
    season_year = "2024-25"
    user_min_acc = "60"
    df = get_game_stats_data_df(league, season_year, training_and_testing=True)
    X, y, *_ = get_X_and_y(df)

    if league == "ncaa":
        X = X[(3 * len(X) // 4):]
        y = y[(3 * len(y) // 4):]

    use_cached_preds_probs = True
    if use_cached_preds_probs:
        with open(
                f"preds_probs/{league}_det_preds_probs_sim_probs.txt", "r"
        ) as f:
            det_preds, det_probs, sim_probs = json.load(f)
    else:
        det_preds, det_probs, sim_probs, CIs = get_all_preds(
            X, y, league, num_sims=1000
        )

        with open(
                f"preds_probs/{league}_det_preds_probs_sim_probs.txt", "w"
        ) as f:
            json.dump([
                np.array(det_preds).tolist(),
                np.array(det_probs).tolist(),
                np.array(sim_probs).tolist()],
                f)

    use_cached_acc_thresholds = True
    if use_cached_acc_thresholds:
        with open(
                f"acc_thresholds/{league}_acc_per_thresholds_home.json", "r"
        ) as f:
            acc_per_thresholds_home = json.load(f)

        with open(
                f"acc_thresholds/{league}_acc_per_thresholds_away.json", "r"
        ) as f:
            acc_per_thresholds_away = json.load(f)
    else:
        acc_per_thresholds_home = get_acc_per_thresholds(
            True, sim_probs, y, verbose=False
            )
        print(acc_per_thresholds_home)
        with open(
                f"acc_thresholds/{league}_acc_per_thresholds_home.json", "w"
        ) as f:
            json.dump(acc_per_thresholds_home, f)

        acc_per_thresholds_away = get_acc_per_thresholds(
            False, sim_probs, y, verbose=False
            )
        print(acc_per_thresholds_away)
        with open(
                f"acc_thresholds/{league}_acc_per_thresholds_away.json", "w"
        ) as f:
            json.dump(acc_per_thresholds_away, f)

    min_threshold = acc_per_thresholds_home[user_min_acc][1]
    sim_preds = (np.array(sim_probs) > min_threshold).astype(int)

    info_dict = display_all_reports(y, det_preds, sim_preds)
    print(info_dict)
    