
import pytest
import pandas as pd
from backend.models.historic_model import get_X_and_y

def test_get_X_and_y():
    df = pd.DataFrame({
        'WL': [1, 0, 1],
        'GAME_DATE': ['2024-01-01'] * 3,
        'TEAM_ID': ['101', '102', '103'],
        'PTS': [90, 85, 88],
        'AST': [20, 19, 21]
    })
    X, y, home_ids, game_dates = get_X_and_y(df)
    assert len(X) == len(y) == len(home_ids) == len(game_dates)
    assert 'PTS' in X.columns and 'AST' in X.columns
