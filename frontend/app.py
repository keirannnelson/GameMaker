import os
import io
import base64
from datetime import datetime, timedelta
import sqlite3

import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file
import firebase_admin
from firebase_admin import credentials, auth 
from backend.models.pred_pipeline import pred_historic_model_old_outcomes_pipeline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

DB_PATH = 'backend/database/game_stats_full.db'
LEAGUE_TO_MODEL_LEAGUE = {'NBA': 'nba', 'NCAAMB_D1': 'ncaa'}

app = Flask(__name__)
app.secret_key = os.environ.get('FIREBASE_SECRET_KEY')

cred = credentials.Certificate('firebase_config.json')
firebase_admin.initialize_app(cred)

# initialize dates
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
seasons = c.fetchall()
season_dates = {}

for season in seasons:
    c.execute(f"SELECT MAX(GAME_DATE), MIN(GAME_DATE) FROM '{season[0]}'")
    end_date, start_date = c.fetchone()
    season_dates[season] = [end_date, start_date]


@app.route('/')
def home():
    return render_template('home.html', user_email=session.get('user_email'))

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')


@app.route('/matchups')
def matchups():
    return render_template('matchups.html')
            
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')
@app.route('/get_games_range', methods=['POST'])
def get_games_range():
    data = request.get_json()
    selected_dates = data.get('selected_dates', [])
    selected_league = data.get('selected_league')

    if not selected_dates:
        return jsonify({'error': 'No dates provided'}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    all_games = {}

    for date_str in selected_dates:
        # Use your existing get_season helper to find the season for this date
        season = get_season(date_str, season_dates)
        if not season:
            all_games[date_str] = []
            continue

        c.execute(f"""
            SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_NAME, WL, MATCHUP, TEAM_ABBREVIATION
            FROM 'game_stats_2024-25'
            WHERE GAME_DATE = ?
            AND LEAGUE = '{selected_league}'
            ORDER BY GAME_DATE ASC
        """, (date_str,))
        rows = c.fetchall()

        games = {}
        for row in rows:
            game_id, game_date, team_id, name, wl, matchup, abbrev = row
            if game_id not in games:
                games[game_id] = []

            if selected_league == 'NCAAMB_D1':
                name = f'{name} ({abbrev})'

            # Determine home team by checking 'vs.' in matchup
            is_home = 'vs.' in matchup
            games[game_id].append({'id': team_id, 'name': name, 'wl': wl, 'home': is_home})

        # Format the games results as you do in /get_games
        results = []
        for game_id, teams in games.items():
            if len(teams) == 2:
                # Set home/away correctly
                if teams[0]['home']:
                    team1 = teams[0]
                    team2 = teams[1]
                else:
                    team1 = teams[1]
                    team2 = teams[0]

                results.append({
                    'id' : game_id,
                    'home': team1['name'],
                    'away': team2['name'],
                    'home_record': get_record(team1['id'], c, date_str, season),
                    'away_record': get_record(team2['id'], c, date_str, season),
                    'home_id': team1['id'],
                    'away_id': team2['id'],
                })
        
        all_games[date_str] = results

    conn.close()
    return jsonify({'games': all_games})


@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/glossary')
def glossary():
    return render_template('glossary.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/sessionLogin', methods=['POST'])
def session_login():
    try:
        id_token = request.json.get('idToken')
        decoded_token = auth.verify_id_token(id_token)
        session['user_email'] = decoded_token.get('email')
        return '', 200
    except Exception as e:
        print("Token verification failed:", e)
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('home'))

# Gets correct season
def get_season(date, season_dates):
    for season in season_dates:
        if date <= season_dates[season][0] and date >= season_dates[season][1]:
            return season[0]
        
    return None

# Count current W-L for each team
def get_record(team_id, c, next_day_str, season):
    c.execute(f"""
        SELECT WL
        FROM '{season}'
        WHERE TEAM_ID = ?
        AND GAME_DATE < ?
    """, (team_id, next_day_str))
    team_games = c.fetchall()
    wins = sum(1 for g in team_games if g[0] == 'W')
    losses = sum(1 for g in team_games if g[0] == 'L')
    return f"{wins}-{losses}"


# gets a list of matchups (dictionaries) for the day of that season
def retrieve_results(season, next_day_str, league):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(f"""
        SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_NAME, WL, MATCHUP, TEAM_ABBREVIATION
        FROM '{season}'
        WHERE GAME_DATE = '{next_day_str}'
        AND LEAGUE = '{league}'
        ORDER BY GAME_DATE ASC
    """)
    rows = c.fetchall()

    games = {}
    for row in rows:
        game_id, game_date, team_id, name, wl, matchup, abbrev = row
        if game_id not in games:
            games[game_id] = []
        
        if league == 'NCAAMB_D1':
            games[game_id].append({'team_id': team_id, 'name': f'{name} ({abbrev})', 'wl': wl, 'home': 'vs.' in matchup})
        else:
            games[game_id].append({'team_id': int(team_id), 'name': name, 'wl': wl, 'home': 'vs.' in matchup})

    team_ids = []
    results = []
    for game_id, teams in games.items():
        if len(teams) == 2:  # Ensure it’s a valid matchup
            # set home/away
            if teams[0]['home']:
                home = teams[0]
                away = teams[1]
            else:
                home = teams[1]
                away = teams[0]

            team_ids += [home['team_id'], away['team_id']]

            
            results.append({
                'home': home['name'],
                'away': away['name'],
                'home_record': get_record(home['team_id'], c, next_day_str, season),
                'away_record': get_record(away['team_id'], c, next_day_str, season),
            })
    conn.close()
    return results, team_ids


@app.route('/get_games', methods=['POST'])
def get_games():
    data = request.get_json()
    selected_date = data.get('selected_dates')  # format: 'YYYY-MM-DD'
    selected_league = data.get('selected_league')
    if not selected_date:
        return jsonify({'error': 'Date not provided'}), 400
    if not selected_league:
        return jsonify({'error': 'league not provided'}), 400

    try:
        selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        next_day = selected_dt + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")
        # Fix get season errors
        season = get_season(next_day_str, season_dates)
        if not season:
            return jsonify({'games': []})

        results, team_ids = retrieve_results(season, next_day_str, selected_league)
        
        return jsonify({'games': results, 'season': season[-7:]})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
    

@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    data = request.get_json()
    selected_date = data.get('selected_date')  # format: 'YYYY-MM-DD'
    selected_league = data.get('selected_league')
    if not selected_date:
        return jsonify({'error': 'Date not provided'}), 400

    try:
        selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        next_day = selected_dt + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")

        season = get_season(next_day_str, season_dates)
        if not season:
            return jsonify({'games': []})

        results, team_ids = retrieve_results(season, next_day_str, selected_league)
        if not results:
            return jsonify({'games': results})
        
        outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = pred_historic_model_old_outcomes_pipeline('deterministic', LEAGUE_TO_MODEL_LEAGUE[selected_league], season[-7:], 60, target_team_ids=team_ids, target_game_date=next_day_str)
        
        for ids, result, in zip(team_ids[::2], results):
            winner = outcomes_preds[f'{next_day_str}:{ids}'][0]
            prediction = outcomes_preds[f'{next_day_str}:{ids}'][1]
            result['winner'] = 'Home' if winner else 'Away'
            result['prediction'] = 'Home' if prediction else 'Away'

        return jsonify({'games': results, 
                        'confusion_matrix': [[int(cms[0][0]), int(cms[0][1])], 
                                             [int(cms[1][0]), int(cms[1][1])]],
                        'season': season[-7:],
                        'stats': {'final_acc': round(accs*100, 1), 'final_recall': round(recalls*100, 2), 'final_precision': round(precisions*100, 2), 'final_f1': round(f1s, 2)}
                        })
    
    except Exception as e:
        print("Error:", e)
        
        return jsonify({'error': str(e)}), 500
    
@app.route("/download_db")
def download_db():
    return send_file(DB_PATH, as_attachment=True)

@app.route('/get_teams')
def get_teams():
    league = request.args.get('league', 'NBA')
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            SELECT DISTINCT TEAM_NAME
            FROM "game_stats_2024-25"
            WHERE LEAGUE = ?
            ORDER BY TEAM_NAME ASC
        """, (league,))
        
        teams = [row[0] for row in c.fetchall()]
        conn.close()
        
        return jsonify({"teams": teams})
    except Exception as e:
        print("Error fetching teams:", e)
        return jsonify({"teams": [], "error": str(e)}), 500
    

TEAM_ABBREVIATION_MAP = {
  "Atlanta Hawks": "ATL",
  "Boston Celtics": "BOS",
  "Brooklyn Nets": "BKN",
  "Charlotte Hornets": "CHA",
  "Chicago Bulls": "CHI",
  "Cleveland Cavaliers": "CLE",
  "Dallas Mavericks": "DAL",
  "Denver Nuggets": "DEN",
  "Detroit Pistons": "DET",
  "Golden State Warriors": "GSW",
  "Houston Rockets": "HOU",
  "Indiana Pacers": "IND",
  "LA Clippers": "LAC",
  "Los Angeles Lakers": "LAL",
  "Memphis Grizzlies": "MEM",
  "Miami Heat": "MIA",
  "Milwaukee Bucks": "MIL",
  "Minnesota Timberwolves": "MIN",
  "New Orleans Pelicans": "NOP",
  "New York Knicks": "NYK",
  "Oklahoma City Thunder": "OKC",
  "Orlando Magic": "ORL",
  "Philadelphia 76ers": "PHI",
  "Phoenix Suns": "PHO",
  "Portland Trail Blazers": "POR",
  "Sacramento Kings": "SAC",
  "San Antonio Spurs": "SAS",
  "Toronto Raptors": "TOR",
  "Utah Jazz": "UTA",
  "Washington Wizards": "WAS",
}

def retrieve_results_matchups(season, next_day_str, team1, team2, league):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if league == 'NBA':
        team1_abbr = TEAM_ABBREVIATION_MAP.get(team1)
        team2_abbr = TEAM_ABBREVIATION_MAP.get(team2)
    else:
        c.execute(f"""SELECT TEAM_NAME, TEAM_ABBREVIATION FROM '{season}' GROUP BY TEAM_ABBREVIATION, TEAM_NAME HAVING LEAGUE != "NBA";""")
        rows = c.fetchall()

# Print table names
        NCAA_map = {}
        for row in rows:
            NCAA_map[row[0]] = row[1]
        team1_abbr = NCAA_map.get(team1)
        team2_abbr = NCAA_map.get(team2)

    if not team1_abbr or not team2_abbr:
        raise ValueError("Invalid team names provided")

    

    c.execute(f"""
        SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_NAME, WL, MATCHUP
        FROM '{season}'
        WHERE GAME_DATE >= ?
        AND LEAGUE = '{league}'
        ORDER BY GAME_DATE ASC
    """, (next_day_str,))
    
    rows = c.fetchall()

    games = {}
    
    for row in rows:
        game_id, game_date, team_id, name, wl, matchup = row

        # Only keep rows that match both abbreviations
        
        if team1_abbr in matchup and team2_abbr in matchup:
            
            if game_id not in games:
                games[game_id] = []
            games[game_id].append({
                'team_id': team_id,
                'name': name,
                'wl': wl,
                'home': 'vs.' in matchup,
                'game_date':game_date
            })

    team_ids = []
    results = []
    for game_id, teams in games.items():
        
        if len(teams) == 2:
            home = teams[0] if teams[0]['home'] else teams[1]
            away = teams[1] if teams[0]['home'] else teams[0]

            team_ids += [home['team_id'], away['team_id']]
            
            results.append({
                'home': home['name'],
                'away': away['name'],
                'home_record': get_record(home['team_id'], c, next_day_str, season),
                'away_record': get_record(away['team_id'], c, next_day_str, season),
                'game_date':teams[-1]['game_date'], 
                'home_id': home['team_id'],
                'away_id':away['team_id'],
                'game_id': game_id
            })


    conn.close()
    return results, team_ids


@app.route('/get_matchups', methods=['POST'])
def get_matchups():
    data = request.get_json()
    selected_date = data.get('selected_date')  # format: 'YYYY-MM-DD'
    team1 = data.get('team1')  # Full name
    team2 = data.get('team2')  # Full name
    selected_league = data.get('selected_league')  # Full name
    
    print(selected_date, team1, team2)
    if not (selected_date and team1 and team2):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        next_day = selected_dt + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")

        season = get_season(next_day_str, season_dates)
        if not season:
            return jsonify({'error': 'Season out of range'}), 400

        results, team_ids = retrieve_results_matchups(season, next_day_str, team1, team2, selected_league)
        return jsonify({'games': results, 'season': season[-7:]})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
        
@app.route('/get_predictions_range', methods=['POST'])
def get_predictions_range():
    data = request.get_json()
    selected_games = data.get('selected_games', [])
    selected_teams = data.get('selected_teams')
    selected_league = data.get('selected_league')
    selected_model = data.get('selected_model')
    accuracy_threshold = data.get('accuracy_threshold')

    if not selected_games or not selected_teams:
        return jsonify({'error': 'No games provided'}), 400

   
    predictions = {}

    dates = [i for i in selected_games.keys()]
    games = []
    for i in selected_games.values():
        games += i

    teams = []
    for i in selected_teams.values():
        if selected_league == 'NBA':
            i = [int(j) for j in i]
        teams += i

    outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = pred_historic_model_old_outcomes_pipeline(
            selected_model, 
            LEAGUE_TO_MODEL_LEAGUE[selected_league], 
            '2024-25', 
            int(accuracy_threshold), 
            target_team_ids=teams, 
            target_game_dates=dates,
            target_game_ids=games)


    predictions = {}
    for key in outcomes_preds:
        outcomes = outcomes_preds[key]
        date, home = key.split(':')
        if date not in predictions:
            predictions[date] = []

        actual_winner = 'Home' if outcomes[0] else 'Away'
        prediction = 'Home' if outcomes[1] else 'Away'
        data = [prediction, actual_winner]
        if selected_model == 'simulation':
            data.append(outcomes[2])
            data.append(outcomes[-1])

        predictions[date].append(data)

    


    return jsonify({'games': predictions,
                    'confusion_matrix': cms.tolist(),
                    'season': season[-7:],
                    'stats': {'final_acc': round(accs*100,2), 'final_recall': round(recalls*100,2), 'final_precision': round(precisions*100,2), 'final_f1': round(f1s,2)}
                    })



@app.route('/test', methods=['POST'])
def test():
    data = request.get_json()
    selected_games = data.get('selected_games', [])
    selected_league = data.get('selected_league')
 
    selected_model = data.get('selected_model')
   
    #changed 
    selected_gameIds = data.get('selected_gameIds', [])
  

    if not selected_games:
        return jsonify({'error': 'No games provided'}), 400

    sum_cm = [[0,0],[0,0]]
    predictions = {}

    if selected_league == 'NBA':   
        league_model = 'nba'
    else:
        league_model = 'ncaa'

    dates = [date for date in selected_games]
    team_ids = []
    for date in selected_games:
        for id in selected_games[date]:
            team_ids.append(id)
    
    
        # if league is NBA cast id to int
    print("____________________________")
    print("version: " + selected_model)

    print("league: " + league_model)

    print(team_ids)

    print(dates)

    print(selected_gameIds)
    print("____________________________")
    result = pred_historic_model_old_outcomes_pipeline(
        version = selected_model, 
        league = league_model,
        season_year = '2024-25', 
        target_team_ids=team_ids, 
        target_game_dates=dates,
        target_game_ids=selected_gameIds
        )
    

       

    outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = result
    for date in selected_games:
        
            
        predictions[date] = []
            
        for ids in selected_games[date][::2]:
            outcomes = outcomes_preds.get(f'{date}:{ids}', None)
            if not outcomes:
                predictions[date].append(['Undefined', 'Undefined'])
                continue

            actual_winner = 'Home' if outcomes[0] else 'Away'
            prediction = 'Home' if outcomes[1] else 'Away'
            data = [prediction, actual_winner]
            if selected_model == 'simulation':
                data.append(outcomes[2])
                data.append(outcomes[-1])

            predictions[date].append(data)

    return jsonify({'games': predictions,
                    'confusion_matrix': cms.tolist(),
                    'season': season[-7:],
                    'stats': {'final_acc': round(accs*100,2), 'final_recall': round(recalls*100,2), 'final_precision': round(precisions*100,2), 'final_f1': round(f1s,2)}
                    })



@app.route('/get_parlay', methods=['POST'])
def get_parlay():
    data = request.get_json()
    selected_date = data.get('selected_date')
    selected_teams = data.get('selected_teams')
    selected_league = data.get('selected_league')
    selected_teams = data.get('selected_teams')
    selected_games = data.get('selected_games')

    if selected_league == 'NBA':
        selected_teams = [int(i) for i in selected_teams]

    outcomes_preds, accs, recalls, precisions, f1s, cms, extra_metrics = pred_historic_model_old_outcomes_pipeline(
        'simulation', 
        LEAGUE_TO_MODEL_LEAGUE[selected_league], 
        '2024-25', 
        60, 
        target_team_ids=selected_teams, 
        target_game_dates=[selected_date],
        target_game_ids=selected_games)
    
    games = []
    prob = 1.0
    for ids in selected_teams[::2]:
        outcomes = outcomes_preds.get(f'{selected_date}:{ids}', None)
        if not outcomes:
            games.append(['Undefined'])
            continue
        games.append('Home' if outcomes[2] > 50 else 'Away')
        

        if outcomes[2] > 50:
            prob = prob * (outcomes[2] / 100)
        else:
            prob = prob * (1 - (outcomes[2] / 100))

    return jsonify({'games': games, 'probability': round(prob*100, 2)})


@app.route('/get_db_table', methods=['POST'])
def get_db_table():
    
    data = request.get_json()
    selected_table = data.get('selected_table')
    league = 'NBA'
    if selected_table == None:
        return jsonify({'error': 'No table provided'}), 400
    elif selected_table == 'teams':
        df = pd.read_sql_table('game_stats_2024-25', f'sqlite:///{DB_PATH}')
        df = df[df['LEAGUE'] == league]
        df = df.drop(['SEASON_ID', 'GAME_ID', 'MATCHUP', 'TEAM_ABBREVIATION', 'LEAGUE', 'GAME_DATE'], axis=1)
        df['WL'] = df['WL'].map({'W': 1, 'L':0})
        team_ids = df["TEAM_ID"].unique()

        teams_data = {}
        for team in team_ids:
            team_data = df[df['TEAM_ID'] == team]
            name = team_data['TEAM_NAME'].unique()[0]
            team_data = team_data.drop(['TEAM_NAME', 'TEAM_ID'], axis=1)
            
            teams_data[team] = [name] + team_data.mean().round(2).tolist()
    
        columns = ['TEAM', 'WL%', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', 'FG3M', 'FG3A', 'FG3%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']
        return jsonify({'teams_data': teams_data, 'columns': columns})
    

    elif selected_table == 'players':
        df = pd.read_sql_table('player_info', 'sqlite:///backend/database/player_info.db')
        df = df[df['IS_ACTIVE'] == True]

        conn = sqlite3.Connection('backend/database/player_career_stats.db')
        c = conn.cursor()
        players_data = {}
        for player in df.itertuples(index=False):
            c.execute(f"""
                SELECT *
                FROM '{player[0]}'
                WHERE SEASON_ID = (SELECT MAX(SEASON_ID) FROM '{player[0]}');
                      """)
            row = c.fetchone()
            
            if not row:
                continue
            recent_season = list(row)[3:]
            players_data[player[0]] = [f'{player[1]} {player[2]}'] + recent_season

        conn.close()
        columns = ['PLAYER', 'TEAM', 'AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG%', 'FG3M', 'FG3A', 'FG3%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF','PTS']

        return jsonify({'teams_data': players_data, 'columns': columns})


def plot_image(plot_data, y_label, team):
    season_year = "2024-25"
    x = np.arange(len(plot_data))
    y = np.array(plot_data)

    # Create plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Games")
    
    if len(x) >= 2:  # Prevent polyfit from failing
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color="red", label="Trend Line")

    ax.set_title(f'{team} {y_label} Trend Over Time ({season_year} Season)')
    ax.set_xlabel('Games Played')
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)

    # Convert to image in-memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

@app.route("/get_plot", methods=['POST'])
def get_plot():
    data = request.get_json()
    table = data.get('selected_table')
    col = data.get('selected_column')
    ids = data.get('row_ids')
    league = 'NBA'

    col_to_stat = {'WL%': 'WL', 'FG%': 'FG_PCT', 'FG3%': 'FG3_PCT', 'FT%': 'FT_PCT', '+/-': 'PLUS_MINUS'}
    if col in col_to_stat:
        stat = col_to_stat[col]
    else:
        stat = col


    df = pd.read_sql_table('game_stats_2024-25', f'sqlite:///{DB_PATH}')
    df = df[df['LEAGUE'] == league]

    images = {}

    for team_id in ids:
        team_df = df[df['TEAM_ID'] == team_id]    
        team_df = team_df.sort_values('GAME_DATE')
        team_df['WL'] = team_df['WL'].map({'W': 1, 'L':0})
        team = team_df['TEAM_NAME'].unique()[0]
        plot_data = team_df[stat].tolist()

        if stat == 'WL':
            for i in range(len(plot_data) - 1):
                plot_data[i+1] += plot_data[i]
            

            for i in range(len(plot_data)):    
                plot_data[i] /= i+1


        images[team] = plot_image(plot_data, col, team)

    

    return jsonify(images)




if __name__ == '__main__':

    debug = True
    if debug:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', port=5000)
