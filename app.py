from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

app=Flask(__name__)
model=pickle.load(open('..\Pnuemonia_Analysis\model_training\models\RandomForest__model.pkl','rb'))

#home page
@app.route('/')
def home():
    return render_template("index.html")

#Preddict page
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    int_features_cleaned=[int(x) if x.isdigit() else 0 for x in int_features[2:]]
    print(int_features_cleaned)
    final_features = [np.array(int_features_cleaned)]
    print('final_features',final_features)
    prediction = model.predict_proba(final_features)
    confidence=dict(zip(model.classes_,prediction[0]*100))
    max_key = max(confidence, key=confidence.get)
    max_value = confidence[max_key]
    result = {max_key: max_value}

    if max_key==1:
        text=f'Pnuemonia Predicted with the confidence of {round(max_value,2)}'
    else:
        text=f'Pnuemonia Not found with the confidence of {round(max_value,2)}'
    return render_template('index.html', prediction_text=text)


if __name__ == '__main__':
    app.run(debug=True)