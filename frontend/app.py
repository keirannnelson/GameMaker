import os
from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file
import firebase_admin
from firebase_admin import credentials, auth 
from backend.models.pipeline import pred_old_outcomes_pipeline

DB_PATH = 'backend/database/game_stats.db'

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
        
        outcomes_preds, final_acc, final_recall, final_precision, final_f1, final_cm = pred_old_outcomes_pipeline(season[-7:], team_ids, next_day_str)
        print(outcomes_preds)
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


if __name__ == '__main__':

    debug = True
    if debug:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', port=5000)