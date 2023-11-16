import datetime
from flask import render_template,request
from app import app,APP_ROOT
from app.utils import load_all_gen, load_all_temps, load_data, load_model, load_scaler, preprocess_to_prediction_sequence

@app.route('/')
def home():
    _, curr_gen = load_data()
    temps, dates = load_all_temps()
    gen = load_all_gen()
    return render_template('index.html',title='Home', current_generation=curr_gen, temps=temps, dates=dates, gen=gen)

@app.route("/predict",methods=["GET","POST"])
def predict():
    try:
        months = int(request.args.get('months'))
    except:
        months = 1
        

    if not request.args.get('from'):
        from_date = str(datetime.date.today())
    else:
        from_date = request.args.get('from')
        
    model = load_model(f'models/model{months}/')
    scaler = load_scaler(f'scalers/scaler{months}.joblib')
    df, curr_gen = load_data(months, from_date)
    prediction = preprocess_to_prediction_sequence(df, model, scaler, months)[0]     

    accuracy_list = [0.052, 0.065, 0.069, 0.079, 0.074, 0.079, 0.085, 0.086, 0.087, 0.093, 0.096, 0.130]
    accuracy = accuracy_list[months - 1] * 100
    temps, dates = load_all_temps(from_date)
    gen = load_all_gen(from_date)
    return render_template('index.html', prediction=prediction, current_generation=curr_gen, accuracy=accuracy, temps=temps, dates=dates, gen=gen, current_selected=months, from_date=from_date)
    

        
    