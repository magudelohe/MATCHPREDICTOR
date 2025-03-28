from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Modelo y columnas
try:
    modelo = joblib.load('modelo_futbol.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
except FileNotFoundError:
    print("Error: No se encontraron los archivos del modelo o las columnas.")
    exit()

equipos_locales = sorted([col.replace('home_team_', '') for col in columnas if col.startswith('home_team_')])
equipos_visitantes = sorted([col.replace('away_team_', '') for col in columnas if col.startswith('away_team_')])

@app.route('/')
def home():
    return render_template('formulario.html', equipos_locales=equipos_locales, equipos_visitantes=equipos_visitantes)

@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    neutral = request.form['neutral'] == 'True'
    goals_home_3 = int(request.form['goals_home_3'])
    goals_away_3 = int(request.form['goals_away_3'])

    entrada = pd.DataFrame([[0]*len(columnas)], columns=columnas)
    col_home = f'home_team_{home_team}'
    col_away = f'away_team_{away_team}'
    if col_home in columnas:
        entrada[col_home] = 1
    if col_away in columnas:
        entrada[col_away] = 1
    if 'neutral' in columnas:
        entrada['neutral'] = int(neutral)
    if 'goals_home_3' in columnas:
        entrada['goals_home_3'] = goals_home_3
    if 'goals_away_3' in columnas:
        entrada['goals_away_3'] = goals_away_3

    resultado = modelo.predict(entrada)[0]
    salida = {0: 'Empate', 1: 'Pierde Local', 2: 'Gana Local'}

    return render_template(
    'formulario.html',
    prediction=salida[resultado],
    equipo_local=home_team,
    equipo_visitante=away_team,
    equipos_locales=equipos_locales,
    equipos_visitantes=equipos_visitantes
)



if __name__ == '__main__':
    app.run(debug=True)