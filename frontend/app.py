import os
from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file
import firebase_admin
from firebase_admin import credentials, auth 
from backend.ml_model.pipeline import pred_old_outcomes_pipeline

NBA_TEAMS = (
            'Atlanta Hawks',
            'Boston Celtics',
            'Brooklyn Nets',
            'Charlotte Hornets',
            'Chicago Bulls',
            'Cleveland Cavaliers',
            'Dallas Mavericks',
            'Denver Nuggets',
            'Detroit Pistons',
            'Golden State Warriors',
            'Houston Rockets',
            'Indiana Pacers',
            'LA Clippers',
            'Los Angeles Lakers',
            'Memphis Grizzlies',
            'Miami Heat',
            'Milwaukee Bucks',
            'Minnesota Timberwolves',
            'New Orleans Pelicans',
            'New York Knicks',
            'Oklahoma City Thunder',
            'Orlando Magic',
            'Philadelphia 76ers',
            'Phoenix Suns',
            'Portland Trail Blazers',
            'Sacramento Kings',
            'San Antonio Spurs',
            'Toronto Raptors',
            'Utah Jazz',
            'Washington Wizards',
        )

DB_PATH = 'backend/database/nba_game_stats.db'

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
    c.execute(f"SELECT MAX(GAME_DATE) FROM '{season[0]}'")
    date = c.fetchone()
    season_dates[date[0]] = season[0]


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
            SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_NAME, WL, MATCHUP
            FROM '{season}'
            WHERE GAME_DATE = ?
            AND TEAM_NAME IN {NBA_TEAMS}
            ORDER BY GAME_DATE ASC
        """, (date_str,))
        rows = c.fetchall()

        games = {}
        for row in rows:
            game_id, game_date, team_id, name, wl, matchup = row
            if game_id not in games:
                games[game_id] = []

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
                    'team1': team1['name'],
                    'team2': team2['name'],
                    'team1_record': get_record(team1['id'], c, date_str, season),
                    'team2_record': get_record(team2['id'], c, date_str, season),
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
    season = None
    for end in season_dates:
        if end < date:
            return season
        
        season = season_dates[end]
        
    return season

# Count current W-L for each team
def get_record(team_id, c, next_day_str, season):
    c.execute(f"""
        SELECT WL
        FROM '?'
        WHERE TEAM_ID = ?
        AND GAME_DATE < ?
    """, (season, team_id, next_day_str))
    team_games = c.fetchall()
    wins = sum(1 for g in team_games if g[0] == 'W')
    losses = sum(1 for g in team_games if g[0] == 'L')
    return f"{wins}-{losses}"


# gets a list of matchups (dictionaries) for the day of that season
def retrieve_results(season, next_day_str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    
    c.execute(f"""
        SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_NAME, WL, MATCHUP
        FROM '{season}'
        WHERE GAME_DATE = '{next_day_str}'
        AND LEAGUE = 'NBA'
        ORDER BY GAME_DATE ASC
    """)
    rows = c.fetchall()

    games = {}
    for row in rows:
        game_id, game_date, team_id, name, wl, matchup = row
        if game_id not in games:
            games[game_id] = []

        
        games[game_id].append({'team_id': team_id, 'name': name, 'wl': wl, 'home': 'vs.' in matchup})

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
    selected_date = data.get('selected_date')  # format: 'YYYY-MM-DD'
    if not selected_date:
        return jsonify({'error': 'Date not provided'}), 400

    try:
        selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        next_day = selected_dt + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")

        season = get_season(next_day_str, season_dates)
        if not season:
            return jsonify({'error': 'Season out of range'})

        results, team_ids = retrieve_results(season, next_day_str)
        return jsonify({'games': results, 'season': season})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
    

@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    data = request.get_json()
    selected_date = data.get('selected_date')  # format: 'YYYY-MM-DD'
    if not selected_date:
        return jsonify({'error': 'Date not provided'}), 400

    try:
        selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        next_day = selected_dt + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")


        season = get_season(next_day_str, season_dates)
        if not season:
            return jsonify({'error': 'Season out of range'})

        results, team_ids = retrieve_results(season, next_day_str)
        outcomes_preds, final_acc, final_recall, final_precision, final_f1, final_cm = pred_old_outcomes_pipeline(season[-7:], team_ids, next_day_str)
        
        for ids, result, in zip(team_ids[::2], results):
            winner, prediction = outcomes_preds[int(ids)]
            result['winner'] = 'Home' if winner else 'Away'
            result['prediction'] = 'Home' if prediction else 'Away'


        return jsonify({'games': results, 
                        'confusion_matrix': [[int(final_cm[0][0]), int(final_cm[0][1])], 
                                             [int(final_cm[1][0]), int(final_cm[1][1])]],
                        'season': season[-7:],
                        'stats': {'final_acc': round(final_acc*100, 1), 'final_recall': round(final_recall*100, 2), 'final_precision': round(final_precision*100, 2), 'final_f1': round(final_f1, 2)}
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
        conn = sqlite3.connect('/Users/alyssaking/Desktop/frontend/game_stats.db')
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

def retrieve_results_matchups(season, next_day_str, team1, team2):
    team1_abbr = TEAM_ABBREVIATION_MAP.get(team1)
    team2_abbr = TEAM_ABBREVIATION_MAP.get(team2)

    if not team1_abbr or not team2_abbr:
        raise ValueError("Invalid team names provided")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(f"""
        SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_NAME, WL, MATCHUP
        FROM '{season}'
        WHERE GAME_DATE = ?
        AND LEAGUE = 'NBA'
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
                'home': 'vs.' in matchup
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
            })

    conn.close()
    return results, team_ids


@app.route('/get_matchups', methods=['POST'])
def get_matchups():
    data = request.get_json()
    selected_date = data.get('selected_date')  # format: 'YYYY-MM-DD'
    team1 = data.get('team1')  # Full name
    team2 = data.get('team2')  # Full name

    if not (selected_date and team1 and team2):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        next_day = selected_dt + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")

        season = get_season(next_day_str, season_dates)
        if not season:
            return jsonify({'error': 'Season out of range'}), 400

        results, team_ids = retrieve_results_matchups(season, next_day_str, team1, team2)
        return jsonify({'games': results, 'season': season})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':

    debug = False
    if debug:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', port=5000)
