from flask import Flask,render_template,request
import numpy as np
import pickle
app=Flask(__name__)
#load the pickle file
model=pickle.load(open('House.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def predict():
    # Retrieve form data
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    location = int(request.form['location'])  # Assuming location is an integer (e.g., encoded value)
    size = int(request.form['size'])
    status = int(request.form['status'])
    facing = int(request.form['facing'])      # Assuming facing is an integer (e.g., encoded value)
    property_type = int(request.form['type']) # Assuming type is an integer (e.g., encoded value)

    # Convert to NumPy array
    data = np.array([[bedrooms, bathrooms, location, size, status, facing, property_type]])

    # Predict the price using the trained model
    predicted_price = model.predict(data)[0]

    # Render the result template and pass the predicted price
    return render_template('result.html', predicted_price=predicted_price)


if __name__ == '__main__':
    app.run(debug=True)