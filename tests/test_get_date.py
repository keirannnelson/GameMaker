import unittest
from frontend.app import get_season

class test_get_season(unittest.TestCase):

    def test_normal_case(self):
        season_dates = {'2025-01-01': 2025, '2024-01-01': 2024, '2023-01-01': 2023}
        self.assertEqual(get_season('2024-12-12', season_dates), 2025)

    def test_none_case(self):
        season_dates = {'2025-01-01': 2025, '2024-01-01': 2024, '2023-01-01': 2023}
        self.assertEqual(get_season('2025-12-12', season_dates), None)
