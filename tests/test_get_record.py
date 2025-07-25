import unittest
import sqlite3

from frontend.app import get_record

class test_get_record(unittest.TestCase):
    
    def setUp(self):
        db_path = '../backend/database/nba_game_stats.db'
        self.connection = sqlite3.Connection(db_path)
        self.cursor = self.connection.cursor()
        
    def tearDown(self):
        self.connection.close()

    def test_valid_record(self):
        team_id = "1610612741"
        next_day_str = "2024-01-01"
        season = "game_stats_2023-24"

        self.assertEqual(get_record(team_id, self.cursor, next_day_str, season), '19-25')

    def test_empty_record(self):
        team_id = "1610612741"
        next_day_str = "2023-01-01"
        season = "game_stats_2023-24"

        self.assertEqual(get_record(team_id, self.cursor, next_day_str, season), '0-0')
