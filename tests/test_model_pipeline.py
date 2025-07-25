import unittest

from backend.ml_model.pipeline import pred_old_outcomes_pipeline

class test_model_pipeline(unittest.TestCase):

    def test_valid_case(self):
        season_year = '2023-24'
        target_team_ids = [1610612755, 1610612741]
        target_game_date = '2024-01-02'
        self.assertIsInstance(pred_old_outcomes_pipeline(season_year, target_team_ids, target_game_date), tuple)


    def test_empty_case(self):
        season_year = '2023-24'
        target_team_ids = [1610612755, 1610612741]
        target_game_date = '2022-01-02'
        self.assertEqual(pred_old_outcomes_pipeline(season_year, target_team_ids, target_game_date), (None, None, None, None, None, None))