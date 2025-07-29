import pandas as pd
import numpy as np
from .historic_model import get_X_and_y, make_preds
from .get_game_stats_data import get_game_stats_data_df
import matplotlib.pyplot as plt
from .monte_carlo_simulation import (
    get_all_preds, display_all_reports, get_sim_probs, CI95_percentage
)
import json
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
)
import os
import joblib


def pred_historic_model_old_outcomes_pipeline(
    version, league, season_year, user_min_acc=60, target_team_ids=None,
    target_game_dates=None, target_game_ids=None, training_and_testing=False
):
    df = get_game_stats_data_df(
        league,
        season_year,
        target_team_ids=target_team_ids,
        target_game_dates=target_game_dates,
        target_game_ids=target_game_ids,
        training_and_testing=training_and_testing,
    )

    X, y, home_ids, game_dates = get_X_and_y(df)

    if version == "all":
        det_preds, det_probs, sim_probs, CIs = get_all_preds(
            X, y, league, num_sims=1000
        )

        if det_preds is None:
            return (None,) * 7
        with open(
                f"backend/models/acc_thresholds/{league}_acc_per_thresholds_home.json", "r"
            ) as f:
            acc_per_thresholds_home = json.load(f)

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
    elif version == "deterministic":
        det_preds, det_probs, acc, recall, precision, f1, cm = make_preds(
            X, y, league,
            verbose=False,
            get_conf_matrix_img=False
        )
        if det_preds is None:
            return (None,) * 7

        outcomes_preds = {
            f"{game_date}:{home_id}": (true_label, det_pred, det_prob) for
            home_id, game_date, true_label, det_pred, det_prob,
            in zip(
                home_ids, game_dates, y, det_preds, det_probs
            )
        }
        acc = accuracy_score(y, det_preds)
        recall = recall_score(y, det_preds, zero_division=0)
        precision = precision_score(y, det_preds, zero_division=0)
        f1 = f1_score(y, det_preds, zero_division=0)
        cm = confusion_matrix(y, det_preds, labels=[1, 0])

        return (
            outcomes_preds, acc, recall, precision, f1, cm, None
        )
    elif version == "simulation":
        folder_path = f'backend/models/{league}_model_fold_data'
        fold_files = [f for f in os.listdir(folder_path) if
                      os.path.isfile(os.path.join(folder_path, f))]
        num_folds = len(fold_files)
        folds = [
            joblib.load(f'{folder_path}/ensemble_fold_{i}.pkl') for i in
            range(num_folds)
        ]
        num_test_cases = X.shape[0]
        sim_probs = [None] * num_test_cases
        CIs = [None] * num_test_cases
        num_sims = 1000
        for i in range(num_test_cases):
            sim_prob = get_sim_probs(
                X, i, folds, num_sims=num_sims
            )
            sim_probs[i] = sim_prob
            CIs[i] = CI95_percentage(sim_prob, num_sims)

        with open(
                f"backend/models/acc_thresholds/{league}_acc_per_thresholds_home.json", "r"
        ) as f:
            acc_per_thresholds_home = json.load(f)

        min_threshold = acc_per_thresholds_home[str(user_min_acc)][1]
        sim_preds = (np.array(sim_probs) > min_threshold).astype(int)

        outcomes_preds = {
            f"{game_date}:{home_id}": (true_label, sim_pred, sim_prob,
                                       sim_pred, sim_prob, CI) for
            home_id, game_date, true_label, sim_pred, sim_prob, CI
            in zip(
                home_ids, game_dates, y, sim_preds, sim_probs, CIs
            )
        }

        acc = accuracy_score(y, sim_preds)
        recall = recall_score(y, sim_preds, zero_division=0)
        precision = precision_score(y, sim_preds, zero_division=0)
        f1 = f1_score(y, sim_preds, zero_division=0)
        cm = confusion_matrix(y, sim_preds, labels=[1, 0])

        return (
            outcomes_preds, acc, recall, precision, f1, cm, None
        )


def trend_line_graph(plot_data, plot_type, league, season_year, version):
    x = range(len(plot_data))
    y = plot_data
    plt.scatter(x, y, label="Data Points")
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color="red", label="Trend Line")
    plt.title(f'{plot_type} Trend Over Time ({season_year} Season)')
    plt.xlabel('Game Days Since First Game Tested')
    plt.ylabel(plot_type)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'backend/models/ml_imgs/{league}_model/{season_year}'
                f'_{version}_{plot_type.lower()}_trend_line.png'
                )
    plt.show()


def eval_model_preds_over_time(version, league, season_year):
    df = pd.read_sql_table(
        f"{league}_game_stats_{season_year}",
        f"sqlite:///backend/database/{league}_game_stats.db"
        )
    print(len(df))
    if league == "nba":
        df = df[df['SEASON_ID'] == f'2{season_year[:season_year.index("-")]}']
    df.sort_values('GAME_DATE', inplace=True)
    if league == "ncaa":
        df = df[(3 * len(df) // 4):]
    print(len(df))
    game_dates = df['GAME_DATE'].unique()
    accs = []
    recalls = []
    precision = []
    f1 = []
    for game_date in game_dates:
        if len(df[df['GAME_DATE'] == game_date]) < 1:
            print(f"Game date {game_date} is empty")

        (outcomes_preds, final_acc, final_recall, final_precision,
         final_f1, final_cm, extra_metrics) =\
            pred_historic_model_old_outcomes_pipeline(
            version, league, season_year, 60, target_game_dates=game_date,
            training_and_testing=True
        )

        if outcomes_preds is None:
            print(f"Had no proper data from {game_date}")
        else:
            accs.append(final_acc)
            recalls.append(final_recall)
            precision.append(final_precision)
            f1.append(final_f1)
            print(game_date)

    trend_line_graph(accs, "Accuracy", league, season_year, version)
    trend_line_graph(recalls, "Recall", league, season_year, version)
    trend_line_graph(precision, "Precision", league, season_year, version)
    trend_line_graph(f1, "F1", league, season_year, version)


if __name__ == '__main__':
    test_nba_pred_old_outcomes_pipeline = False
    test_range_preds = True
    test_ncaa_pred_old_outcomes_pipeline = False
    do_eval_model_over_time = False

    def print_results(
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
    ):
        print("=" * 100)
        if outcomes_preds:
            print(outcomes_preds)
            print(f"Accuracy: {accs}")
            print(f"Recall: {recalls}")
            print(f"Precision: {precisions}")
            print(f"F1 Score: {f1s}")
            print("Confusion matrix:")
            print(cms)
            print(extra_metrics)
        else:
            print("No predictions could be made")
        print("=" * 100)

    if test_nba_pred_old_outcomes_pipeline:
        version = "all"
        league = "nba"
        season_year = "2024-25"
        user_min_acc = 50
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                version, league, season_year, user_min_acc,
                target_team_ids=[
                    1610612747, 1610612757, 1610612744, 1610612755,
                    1610612760, 1610612746, 1610612750, 1610612738,
                    1610612749, 1610612751, 1610612748, 1610612754
                ],
                target_game_dates="2025-01-02",
                training_and_testing=True
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )
    if test_range_preds:
        version = "deterministic"
        league = "nba"
        season_year = "2024-25"
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                version, league, season_year,
                target_team_ids=[1610612760, 1610612737],
                target_game_dates=["2024-10-17"],
                target_game_ids=['0012400064'],
                training_and_testing=False
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                version, league, season_year,
                target_team_ids=[1610612751, 1610612737],
                target_game_dates=["2024-10-23"],
                target_game_ids=['0022400064'],
                training_and_testing=False
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                version, league, season_year,
                target_team_ids=[1610612766, 1610612745],
                target_game_dates=["2024-10-23"],
                target_game_ids=['0022400068'],
                training_and_testing=False
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                version, league, season_year,
                target_team_ids=[1610612766, 1610612737],
                target_game_dates=["2024-10-25"],
                target_game_ids=['0022400079'],
                training_and_testing=False
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                version, league, season_year,
                target_team_ids=[1610612766, 1610612737, 1610612751,
                                 1610612760, 1610612766, 1610612745],
                target_game_dates=["2024-10-17", "2024-10-23", "2024-10-25"],
                target_game_ids=['0012400064', '0022400064', '0022400079', '0022400068'],
                training_and_testing=False
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )

    if test_ncaa_pred_old_outcomes_pipeline:
        version = "deterministic"
        league = "ncaa"
        season_year = "2024-25"
        user_min_acc = 60
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = (
            pred_historic_model_old_outcomes_pipeline(
                version, league, season_year, user_min_acc,
                target_team_ids=[
                    "912f8837-1d81-4ef9-a576-a21f271d4c64",
                    "0113eea0-c943-4fff-9780-ae0fb099e7ef",
                    "f7575278-12ec-494a-b544-427c1759d43d",
                    "16de38c6-062f-48b1-848f-9c4a8ce7cfa8",
                    "5fe4a398-1699-4442-a364-305b2f0ac01e",
                    "1e42cce3-236b-4f3e-9650-2c45e6332e1e",
                    "632616c5-2dbb-4017-a449-c9dfc303f026",
                    "0b73b9b9-1e29-4b14-b5d8-0405fe942e32",
                    "e66d7ed2-9b4c-48c8-adaa-a276793b057d",
                    "ca74bd67-034c-4468-8487-6f983f5eb4f6",
                    "5c7bf63f-bc39-43c5-9907-73b50b7a6b34",
                    "7a270923-01d6-4a6f-a93e-e8786266d502",
                    "255c2a87-40c0-4989-bbb9-d286aac5d28e",
                    "e6c94452-1adf-4fb9-90e5-10f165c57c5a",
                    "558abf02-97f4-4323-aa99-63c076d35932",
                    "88ff8c00-958e-4ccf-a21d-77fab9e93692",
                    "d60357bd-1205-42e9-9092-d986a2843a34",
                    "6a8d2a24-2128-47a0-913f-3d29e9d6592c",
                    "a381465b-4530-4a00-a50c-5253cd7695a0",
                    "3da4798c-2d58-4dac-b66e-ad145a91b544",
                    "56fe0ab2-e4f0-47b9-8726-9ce23ebcde20",
                    "12d7f888-675b-459f-9099-a38f771d8a95",
                    "c3f0a8ce-af67-497f-a750-3b859376b20a",
                    "15d31915-fbd6-4ae3-8e4b-f3b563c56a18",
                    "aba39ad0-6d9f-4946-bf7e-5c63bf966ce6",
                    "fa416692-7e09-4f0a-9bcf-0cf7d5149a14",
                    "e52c9644-717a-46f4-bf16-aeca000b3b44",
                    "c1f4aae1-aa16-4095-aeab-10e5c2a1236a",
                    "7d3b9c72-cbdd-4b64-9aa3-a5bde9b75fdc",
                    "b827dbdf-230f-4916-9703-3bb9004e26eb",
                    "4fbebf0a-e117-4a0c-8f15-c247535a2a1b",
                    "24051034-96bb-4f78-a3a6-312f3258780f",
                    "22d90601-19d4-461b-a901-924d12f116ed",
                    "7888d144-563d-4f84-b99b-d120aecf1fef",
                    "3b644902-cd06-4930-9a3e-6c78cfb1f464",
                    "a7086972-971c-4333-9944-5120c6ed6c42",
                    "32755362-6336-4dd3-ac34-6f72a6cc150b",
                    "4927ef8b-e680-4fb8-b10e-2ea3eff32eab",
                    "f2164f01-2cbc-4967-bb1c-93656d790921",
                    "2ed86226-3fff-4083-90fa-6b02307e6f73",
                    "a3676da1-8404-4065-88a8-3e6d38f847c5",
                    "d12b20ce-8d80-4085-9b77-0373b2ee7d5e",
                    "4383eb6a-7fd8-4ff2-94c5-43c933121e88",
                    "7d797407-623e-476d-b299-46de4275414d",
                    "87d88c93-29bc-4306-b0a6-87c6f6e80da1",
                    "94ea835b-8ce9-4de2-a5b1-971548fea974",
                    "ad4bc983-8d2e-4e6f-a8f9-80840a786c64",
                    "9b166a3f-e64b-4825-bb6b-92c6f0418263",
                    "5326c079-3dc3-468f-a601-b776dd9c1e62",
                    "394d1b2b-7a19-4d43-b04a-5f366c24e5bf",
                    "1b78e7f6-f25c-4921-98e2-9bc565f8dfb4",
                    "61a3e5ab-1be3-4694-b83f-edae0953f409",
                    "6e5dfe2c-28cc-411f-b846-af8436093ab2",
                    "36ca2008-cd2e-4549-8b3b-c745167e07f3",
                    "a1ba4b89-d97c-44e2-835f-79ad3ccaa5ae",
                    "77ca152a-cb2f-48a5-97b2-492351250d94",
                    "412e7f0d-fb93-4e6e-be1f-9a4c7490d121",
                    "c13f96ba-c79c-452f-9f63-3b45fae4e543",
                    "52df1e19-b142-4a76-a439-ad68455d0581",
                    "dcf5c2e7-c227-4c20-af26-715d5f859412",
                    "4b3ff02c-e0ba-435b-a565-6075bc491684",
                    "d3c8a767-b6fc-45d4-a5f2-18406f2993f1",
                    "0e4258fa-3154-4c16-b693-adecab184c6c",
                    "d203f38a-a166-4258-bca2-e161b591ecfb",
                    "80a19b57-d6ab-4bee-934f-33a72c2e958a",
                    "6af02334-9ac3-452f-bcf4-20abdb72bd07",
                    "b06b63b0-20e6-4c78-bdcf-5feac3d7995e",
                    "a311188e-1259-4fda-b0ea-c47cb52694b1",
                    "51a19d36-24f3-442f-8777-e2e35f5e03b6",
                    "a93bf4f2-3724-4539-8c8a-9ac8ee3741a6",
                    "63a8fdcf-b51b-4de3-8a57-2a6e41d362ce",
                    "8ddbfca9-a931-4908-aa31-9fcd17624b5f",
                    "7cdfcf92-7fe7-46f1-9b66-6912114182e6",
                    "aed211c3-23a4-4188-ad70-22c6eba7d765",
                    "3b088563-ca8a-49a6-9e19-5e4cb186ae1d",
                    "98d2123a-c7b9-4b5f-8a81-43f76d8537dc",
                    "18e89867-9201-41ce-ba19-fadddf92fa17",
                    "0095032d-6143-44f2-8974-f6815fc56c5b",
                    "ca441726-0c57-4456-b1e7-fff098484fb5",
                    "79871449-f5a1-4650-b3cf-85b78fd6943e",
                    "f2cf9ba2-ac8c-4d92-8e7a-48dd88e8f40b",
                    "2e78bd1b-9422-491e-9c73-37cf83b0e34e",
                    "da7d41bc-48b0-4a04-948f-792d6470bcb5",
                    "90ec10eb-38a4-4a69-b072-ef71d294933b",
                    "61a3908a-7492-4b6f-809e-12c61976bb0a",
                    "f9e4261e-d11d-46c4-bd33-c7bbc94ef0e8",
                    "fea46ac5-6dad-43cd-a770-75554dbcc118",
                    "88104678-e53b-43b3-82f7-efb3a11cedb9",
                    "2f513e67-f019-4eec-9096-fbc5911858b6",
                    "6dfaf0ba-47c4-4e05-b0a7-72734747d48f",
                    "d7d668ec-edaa-4d6b-be25-2ebba4128643",
                    "0c0608b3-f349-4f5e-9a10-7e6a744dd0d2",
                    "3db7336c-c18a-441b-912e-e2a4408f12ea",
                    "0affc15f-641d-4211-970d-fb9fb8d36842",
                    "8598d1b6-106b-4083-a512-2e495729525a",
                    "a601205f-3bfe-4052-ac0b-b8f18cf3efd3",
                    "34dd83ac-64f2-4b73-8577-899b2a46f5cc",
                    "3f0ac618-3742-471a-9d1f-03e08f3deca2",
                    "2f8c85a7-54f4-4942-be16-2cbac77c82d7",
                    "b5a804f1-19a4-4183-ab05-43e309e65a5a",
                    "500d7223-ecff-494b-9539-28427156c783",
                    "61dadefa-76bf-4db4-8067-f88df540b9cd",
                    "175d34a7-3823-4e4f-9f11-2464f55360b8",
                    "de9a0e19-b83c-4b98-a6da-81c110cb364b",
                    "de8fc8a7-253f-4597-8a48-a0104ef226ae",
                    "6ef40534-5fad-4755-84de-7dcbd645d1f0",
                    "4b7dedc0-7b48-49a4-aad6-8a94a33274d2",
                    "c7569eae-5b93-4197-b204-6f3a62146b25",
                    "4c9fb59b-6cec-4b0d-bb0f-628b391d138c",
                    "a14b0057-8eb5-43d2-a33b-666196da933e",
                    "a3fef707-e4fc-48a0-82f1-f4ef01b76f45",
                    "600aad0e-85e7-4db8-be0b-94ba02a08e55",
                    "ad9a3e21-4585-4e35-8946-290fe4a16f18",
                    "d52c3640-069c-4554-982e-e6537c8044f1",
                    "3d2f0819-8f3b-4ff3-afd6-6227c469a751",
                    "06d15d35-4955-4fdd-83d9-32d24dbd795b",
                    "22fbe5dc-c6db-4675-b3a3-2d8af7f5d313",
                    "161354af-1f3e-4d58-88f7-e016ec74b7b6",
                    "7c2b372b-b5f0-466e-90cf-880beec57584",
                    "ddfc1dde-4475-4707-921c-f70804422573",
                    "881380da-861c-4ab6-a7d2-20699d8ea883",
                    "5016fe1a-9571-4d10-bf5b-b9c1b496bd57",
                    "dfe0d93f-94a5-47fb-b7aa-f74786e09acb",
                    "2a997096-a381-469d-a7e9-9e031c8b071c",
                    "8ab00d43-840a-4c96-bdee-bf88fa6e3f11",
                    "0d037a5d-827a-44dd-8b70-57603d671d5d",
                    "b83de2df-328f-4303-9895-cdff048fb1ed",
                    "77a69fb0-1355-4342-ac09-b4cc7949d95e",
                    "5ef64f01-86ae-4553-9834-c79cc0859eaf",
                    "eda6c307-1e0f-4a28-82f4-ee9653b343fb",
                    "4743cb7c-784a-4b95-a380-5471f92f2217",
                    "54df21af-8f65-42fc-bc01-8bf750856d70",
                    "f1f4f2a9-db49-4d80-91b9-65e7965082c9",
                    "71874e7e-8260-43f9-bb7c-65f267dbe8ce",
                    "1165ca31-f181-4206-b727-c4e897e4b5cd",
                    "327f09e2-e75f-4014-8ef7-caf9202cd583",
                    "b03bb029-4499-4a2c-9074-5071ed360b21",
                    "bdc2561d-f603-4fab-a262-f1d2af462277",
                    "eb157f98-0697-459c-9293-ddb162ceb28b",
                    "91ec9ba2-12e5-4a75-9410-a166c82163fd",
                    "6bff595c-32cb-4028-837c-c7de2a2107e6",
                    "227d3567-d57b-417e-afc4-735e4cb308a1",
                    "b2fda957-e15c-4fb2-8a13-6e58496f561e",
                    "72971b77-1d35-40b3-bb63-4c5b29f3d22b",
                    "b71d5a1b-2671-4e5a-b94b-06bfb22a27dd",
                    "d6f7e93d-66ec-450d-8fd2-5571d36f7d7f",
                    "cc9e5424-ab3b-41ec-b2aa-7e8a73d759c6",
                    "c206712f-2298-4279-b76c-834ac42809b9",
                    "0f63a6f5-bda7-4fd9-9271-8d33f555ca19",
                    "b795ddbc-baab-4499-8803-52e8608520ab",
                    "39d00ef7-320d-47cf-a2fa-859bc24d16b4",
                    "8aaad1c0-d16e-4b9f-8541-dac670addd71",
                    "576b816b-5b9a-4768-a62b-f8435a8272c2",
                    "d20216d8-161c-46e3-b8d2-92208ddf5acf",
                    "aeaaef0d-5238-414e-ac04-c55a22cba208",
                    "ab9a1315-293f-42d3-a164-860216e81576",
                    "e48e3dd5-721c-4d96-8ff4-790fe2597bfd",
                    "a0fdb660-a822-4337-b01d-31c4d2c99c8a",
                    "c2104cdc-c83d-40d2-a3cd-df986e29f5d3",
                    "4f4b0771-994c-4126-822d-7525aaa00f65",
                    "8ff733a9-9f93-46de-ba7e-cf6f7e429670",
                    "67322042-9c40-4dc2-b33a-4754c02ec82a",
                    "c1c1e6df-a383-4fbd-ba7b-32d4f9ef9518",
                    "9b66e1e0-aace-4671-9be2-54c8acf5ecfc",
                    "c1c54dbf-805c-47fc-a707-c803e94db2a4",
                    "dd8db4d8-d984-4cab-b7f6-22c6b8c2c45f",
                    "56913910-87f7-4ad7-ae3b-5cd9fb218fd9",
                    "324ccccc-575b-4b82-acb0-fbb8e68da13b",
                    "58d8baa3-7624-4b21-a47f-a23df2bf8859",
                    "4af63ebd-d3c8-4772-bbde-938a078bd057",
                    "c0d19efd-d40c-4e32-8e90-dbcb28178b5d",
                    "d2ef641c-70e4-48bb-b40d-ac654d179205",
                    "6a67ba19-56a8-4dd8-a5ae-9e9f2523c274",
                    "ec6185b7-4e0c-4eb8-99ef-f3a4dccf6b91",
                    "db6e1cab-3fa3-4a93-a673-8b2a358ff4bf",
                    "fae4855b-1b64-4b40-a632-9ed345e1e952",
                    "054c3e85-0552-4549-b123-7e84af6e7b6c",
                    "c31455b2-8a45-4248-aa8f-ce7eab1c6b02",
                    "7e42fa84-68cd-47a6-b49a-18b5414d8084",
                    "1753768d-e46e-40b1-8d69-a8ae5cccec03",
                    "524e3eec-7dde-45c7-b3cc-6308cec73350",
                    "ce967953-5c50-4220-87b2-99acb9606e84",
                    "2920c5fa-1e86-4958-a7c4-1e97b8e201d8",
                    "e76479d0-a768-46d2-89bf-7b9dae12bec8",
                    "a8f75c12-c4db-401f-99e3-b48209d85274",
                    "a44f8c85-7ab6-4a4e-b6f1-c69a4cabe1ed",
                    "3a000455-de7c-4ca8-880e-abdce7f21da9",
                    "a41d5a05-4c11-4171-a57e-e7a1ea325a6d",
                    "6120ac5d-e0f2-4b28-9e6f-ee71596a7e88",
                    "6b955b96-b736-475e-bffd-e4acf2054169",
                    "55cd5bb4-b030-4a7a-9652-6cebaf81c574",
                    "e7ff6d5c-07e9-42af-955d-0a0b8c1e2288",
                    "31aedd91-a77e-46c1-8bdc-80e9860c159d",
                    "508f503f-dc57-4f4c-a01e-4f195e5c05c8",
                    "1cb7d0f5-6529-4b71-80a5-a2bee1d505a8",
                    "23fec4e7-108f-4569-85d1-bef2f794e925",
                    "95c92bbf-3fbe-471f-b7bc-375f9018d785",
                    "fe037cba-25d9-42f5-a461-61d7102c17de",
                    "5f9a90a2-3926-404e-8c8a-d9f22ad6907d",
                    "ffa04cde-aa2d-4e91-b2cd-bde2bfed44de",
                    "6373b18c-62f6-49bc-bd4c-8959a2466516",
                    "18585f21-1d63-4400-974c-433fd5073c34",
                    "6a7083ab-1832-48c0-9168-427b35adbcde",
                    "87721c44-53a2-47aa-9b3a-0f1c99b0f328",
                    "fe406882-9f22-495e-9df6-ef357a6803c6",
                    "558abf02-97f4-4323-aa99-63c076d35932",
                    "a3b13d27-ec14-4a7d-88a2-0bc9f7b58984",
                    "1bddc811-44cf-4e53-b4ab-11c8dc82c9de",
                    "509aacc4-c121-46a2-800a-d5431ef181d2",
                    "df3da9d7-2d2b-44e2-99e2-08c4945a203b",
                    "a2b8223d-40b3-4076-b5df-55655c2f8591",
                    "427da272-8ce3-45db-ab29-c612dc74c1e0",
                    "29de4e5e-ee5b-48c8-bffc-1972e7da1e30",
                    "aa7af640-5762-4686-9181-39f7b8a8186e",
                    "1e5dc9e4-c60d-487b-8279-dfbec9229faa",
                    "95826e36-ea33-4b51-83a8-7b4cc20999ee",
                    "a9f5c9a2-342c-4881-9996-5d5c05ec9bd9",
                    "315c4deb-3c9d-413d-a369-7168f2545d21",
                    "2e8442e9-bd09-4d46-9dd3-de79433287f3",
                    "a8ac24ca-3b7a-4610-ad99-042661584c48",
                    "13358462-e05b-4449-8688-90c4622cdde8",
                    "559db90c-741f-40d1-aa81-3fdd1d4889f3",
                    "88ff8c00-958e-4ccf-a21d-77fab9e93692",
                    "5c7bf63f-bc39-43c5-9907-73b50b7a6b34",
                    "92456e1e-f48c-4385-96ab-14cb27d18726",
                    "fe21a988-3e96-4f3c-8f9f-b449ccda43b0",
                    "0796c3b6-5308-4aa0-8fdc-c65d39b043c6",
                    "267d417a-8f85-4c87-a15a-068c089a74c6",
                    "6776c7e6-f4c2-47bd-99e5-39fc179a3197",
                    "c444e03b-d16b-48b2-9b18-823c2647d41d",
                    "0dadedb0-2bd3-45e7-91a3-93af6c4e87f2",
                    "9febd5fa-982f-4f40-ad1d-8e49be96cf4f",
                    "912671c7-19fd-451b-813e-885485427820",
                    "95a27dc5-16d5-4745-9adb-34c41e1444e8",
                    "ca74bd67-034c-4468-8487-6f983f5eb4f6",
                    "d60357bd-1205-42e9-9092-d986a2843a34",
                    "3f4515ea-4472-46b9-ac9e-16d7d8f22a38",
                    "683ab61f-546f-44da-b085-c3a5740554aa",
                    "620d5944-7156-47d3-aad5-5b3824557d03",
                    "441a11b4-b506-45b3-8030-fe72a4381c40",
                    "10f450ce-143b-4e46-8bfe-52c644b68b07",
                    "953151f2-1e42-4cb0-a99e-202493a64dcf",
                    "1f99a164-d593-4d81-85d5-0d7889d6f486",
                    "f8c705b7-87d3-411a-9c8b-5124daab0469",
                    "cec2527e-5e1e-4817-a628-35666ef13b6e",
                    "949c3398-85e4-4c63-ba71-9a82e06ddea4",
                    "fb28bd56-9e56-40c1-992d-71c1e27fe4fd",
                    "09920a5f-1b25-466c-b5ae-6167214f5ba9",
                    "612f7f66-1de1-4d42-b842-9a508daab911",
                    "9916713f-b243-452b-93ae-a2c3ccabf68b",
                    "6f3eec09-2918-4739-a0e8-5d79e14a8332",
                    "f2d01b77-0f5d-4574-9e49-2a3eaf822e44",
                    "e3848467-66c0-41e5-8283-02e3586d8601",
                    "1dc13b18-f9b3-4bb9-b1cf-979fcd8c2b6f",
                    "e6c94452-1adf-4fb9-90e5-10f165c57c5a",
                    "ca478771-aa3d-4231-81e0-b70f519134fb",
                    "70e2bedd-3a0a-479c-ac99-e3f58aa6824b",
                    "6ed15092-2670-450a-99c2-61d861e87644",
                    "faeb1160-5d15-4f26-99fc-c441cf21fc7f",
                    "e9ca48b2-00ba-41c0-a02b-6885a2da1ff1",
                    "1394dd8a-040e-4509-9ee3-761d60eaf6c9",
                    "5873529e-e5e3-4a06-8a03-fa4cbe509880"
                ],
                target_game_dates="2025-02-01",
                training_and_testing=True
            )
        )
        print_results(
            outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics
        )

    if do_eval_model_over_time:
        version = "deterministic"
        league = "ncaa"
        season_year = "2024-25"
        eval_model_preds_over_time(version, league, season_year)
