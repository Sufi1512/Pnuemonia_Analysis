from flask import Flask,render_template,jsonify
import pickle
import numpy as np
import requests
app=Flask(__name__)
model=pickle.load(open('..\Pnuemonia_Analysis\model_training\models\RandomForest__model.pkl','rb'))

@app.route('/detect',methods=['POST'])


def query(filename):
    API_URL = "https://api-inference.huggingface.co/models/nickmuchi/vit-finetuned-chest-xray-pneumonia"
    headers = {"Authorization": "Bearer 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"}
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("a.png")


#home
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))



if __name__ == '__main__':
    app.run(debug=True)