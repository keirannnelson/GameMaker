from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file
import firebase_admin
from firebase_admin import credentials, auth 
from backend.models.pipeline import pred_old_outcomes_pipeline

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
            'South Bay Lakers',
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

DB_PATH = 'backend/database/game_stats.db'

app = Flask(__name__)
app.secret_key = '2e354a049a01caa6d1b91438f1bfb660f8bceb28c13e28e5e40dc8c8a27233eb'  

cred = credentials.Certificate('frontend/firebase_config.json')  
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
        FROM '{season}'
        WHERE TEAM_ID = ?
        AND GAME_DATE < ?
    """, (team_id, next_day_str))
    team_games = c.fetchall()
    wins = sum(1 for g in team_games if g[0] == 'W')
    losses = sum(1 for g in team_games if g[0] == 'L')
    return f"{wins}-{losses}"

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

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        
        c.execute(f"""
            SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_NAME, WL, MATCHUP
            FROM '{season}'
            WHERE GAME_DATE = '{next_day_str}'
            AND TEAM_NAME IN {NBA_TEAMS}
            ORDER BY GAME_DATE ASC
        """)
        rows = c.fetchall()

        games = {}
        for row in rows:
            game_id, game_date, abbrev, name, wl, matchup = row
            if game_id not in games:
                games[game_id] = []

            
            games[game_id].append({'abbrev': abbrev, 'name': name, 'wl': wl, 'home': 'vs.' in matchup})


        results = []
        for game_id, teams in games.items():
            if len(teams) == 2:  # Ensure it’s a valid matchup
                # set home/away
                if teams[0]['home']:
                    team1 = teams[0]
                    team2 = teams[1]
                else:
                    team1 = teams[1]
                    team2 = teams[0]
                

                
                results.append({
                    'team1': team1['name'],
                    'team2': team2['name'],
                    'team1_record': get_record(team1['abbrev'], c, next_day_str, season),
                    'team2_record': get_record(team2['abbrev'], c, next_day_str, season),
                })

            #if len(results) == 100:
            #    break
        
        conn.close()
        print(results)
        return jsonify({'games': results, 'season': season[-7:]})

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

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        

        c.execute(f"""
            SELECT GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, WL, MATCHUP
            FROM '{season}'
            WHERE GAME_DATE = '{next_day_str}'
            AND TEAM_NAME IN {NBA_TEAMS}
            ORDER BY GAME_DATE ASC
        """)
        rows = c.fetchall()
        

        games = {}
        for row in rows:
            game_id, game_date, team_id, abbrev, name, wl, matchup = row
            if game_id not in games:
                games[game_id] = []

            games[game_id].append({'id': team_id, 'abbrev': abbrev, 'name': name, 'wl': wl, 'home': 'vs.' in matchup})

        results = []

        team_ids = []

        for game_id, teams in games.items():
            if len(teams) == 2:  # Ensure it’s a valid matchup
                # set home/away
                if teams[0]['home']:
                    team1 = teams[0]
                    team2 = teams[1]
                else:
                    team1 = teams[1]
                    team2 = teams[0]
                
                team_ids += [team1['id'], team2['id']]

                results.append({
                    'team1': team1['name'],
                    'team2': team2['name'],
                    'team1_record': get_record(team1['id'], c, next_day_str, season),
                    'team2_record': get_record(team2['id'], c, next_day_str, season),
                })
        conn.close()
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


if __name__ == '__main__':
    app.run(debug=True)

