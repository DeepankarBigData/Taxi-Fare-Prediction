import numpy as np
from flask import Flask, request, render_template
from geopy import Nominatim
import joblib as joblib

model = joblib.load('model.pkl')
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global pickup_latitude, dropoff_latitude, pickup_longitude, dropoff_longitude

    address1 = request.form['pickup']
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(address1)

    if location is not None:
        pickup_longitude = location.longitude
        pickup_latitude = location.latitude
    else:
        print("Location not found.")

    address2 = request.form['dropout']
    location = geolocator.geocode(address2)

    if location is not None:
        dropoff_longitude = location.longitude
        dropoff_latitude = location.latitude
    else:
        print("Location not found.")

    def distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
        p = 0.017453292519943295  # Pi/180
        a = 0.5 - np.cos((dropoff_latitude - pickup_latitude) * p) / 2 + np.cos(pickup_latitude * p) * np.cos(
            dropoff_latitude * p) * (1 - np.cos((dropoff_longitude - pickup_longitude) * p)) / 2
        return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

    var_1 = request.form['passenger_count']

    var_6 = 1.609 * distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

    inputs = [pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, var_1, var_6]

    inputs = [float(i) for i in inputs]

    prediction = model.predict([inputs])
    # output = round(prediction[0], 2)
    return render_template("index.html",
                           prediction_text=" The cost of your  journey will be {} dollars".format(prediction))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
