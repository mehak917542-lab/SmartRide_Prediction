from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

#Load model pipeline (includes preprocessor)
with open('gradient_model.pkl', 'rb') as f:
    model_pipeline = joblib.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(" Received POST request")
        print("Form data:", request.form)

        #Collect input
        data = {
            'car_name': request.form['car_name'],
            'brand': request.form['brand'],
            'model': request.form['model'],
            'vehicle_age': int(request.form['vehicle_age']),
            'km_driven': int(request.form['km_driven']),
            'seller_type': request.form['seller_type'],
            'fuel_type': request.form['fuel_type'],
            'transmission_type': request.form['transmission_type'],
            'mileage': float(request.form['mileage']),
            'engine': int(request.form['engine']),
            'max_power': float(request.form['max_power']),
            'seats': int(request.form['seats'])
        }

        #Create DataFrame
        input_df = pd.DataFrame([data])

        #Predict using pipeline
        prediction = model_pipeline.predict(input_df)[0]

        return render_template('result.html', prediction=round(prediction, 2))
    
    except Exception as e:
        print("Error:", e)
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)








