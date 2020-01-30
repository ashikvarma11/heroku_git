import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = model.predict(final_features)
        output = round(output[0],2)
    except:
        return render_template('index.html', prediction_text="You bloody bedsheet. Nere chowe number type cheyada.. Sed aakkale mone!!") 
    return render_template('index.html', prediction_text="Your salary should be ${}".format(output))

if __name__=='__main__':
    app.run(debug=True)