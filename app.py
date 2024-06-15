from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Loading the trained model
model_filename = 'lr_model_co2_emission.pkl'
model = joblib.load(model_filename)


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data and convert to DataFrame
    form_data = request.form.to_dict()

# Columns in same sequential ordwr as of the model
    columns = ['Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders', 'Transmission',
               'Fuel Type', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)',
               'Fuel Consumption Comb (L/100 km)']

    input_data = pd.DataFrame([form_data], columns=columns)


    # Convert numeric fields to float types
    input_data['Engine Size(L)'] = input_data['Engine Size(L)'].astype(float)
    input_data['Cylinders'] = input_data['Cylinders'].astype(int)
    input_data['Fuel Consumption City (L/100 km)'] = input_data['Fuel Consumption City (L/100 km)'].astype(float)
    input_data['Fuel Consumption Hwy (L/100 km)'] = input_data['Fuel Consumption Hwy (L/100 km)'].astype(float)
    input_data['Fuel Consumption Comb (L/100 km)'] = input_data['Fuel Consumption Comb (L/100 km)'].astype(float)



    # Prediction making
    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        return str(e)

    return render_template('result.html', prediction=prediction)




if __name__ == '__main__':
    app.run(debug=True)
