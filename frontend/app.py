from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import firebase_admin
from firebase_admin import credentials, auth



app = Flask(__name__)
app.secret_key = '2e354a049a01caa6d1b91438f1bfb660f8bceb28c13e28e5e40dc8c8a27233eb'  

cred = credentials.Certificate('firebase_config.json')  
firebase_admin.initialize_app(cred)

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




# Count current W-L for each team
def get_record(team_abbrev, c, next_day_str):
    c.execute("""
        SELECT WL
        FROM game_stats
        WHERE TEAM_ABBREVIATION = ?
        AND GAME_DATE < ?
    """, (team_abbrev, next_day_str))
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

        conn = sqlite3.connect('backend/database/game_stats.db')
        c = conn.cursor()

        nba_teams = (
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

        c.execute(f"""
            SELECT GAME_ID, GAME_DATE, TEAM_ABBREVIATION, TEAM_NAME, WL
            FROM game_stats
            WHERE GAME_DATE >= ?
            AND TEAM_NAME IN {nba_teams}
            ORDER BY GAME_DATE ASC
        """, (next_day_str,))
        
        rows = c.fetchall()

        games = {}
        for row in rows:
            game_id, game_date, abbrev, name, wl = row
            if game_id not in games:
                games[game_id] = []
            games[game_id].append({'abbrev': abbrev, 'name': name, 'wl': wl})

        results = []
        for game_id, teams in games.items():
            if len(teams) == 2:  # Ensure it’s a valid matchup
                team1 = teams[0]
                team2 = teams[1]

                

                results.append({
                    'team1': team1['name'],
                    'team2': team2['name'],
                    'team1_record': get_record(team1['abbrev'], c, next_day_str),
                    'team2_record': get_record(team2['abbrev'], c, next_day_str),
                })

            if len(results) == 10:
                break

        conn.close()
        return jsonify({'games': results})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

