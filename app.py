from flask import Flask, render_template, request
import pandas as pd
from sklearn.externals import joblib
import spam_classifier
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('home.html')

@app.route("/predict", methods = ['POST'])
def predict():
    tfidf_model = open("tfidf.pkl", "rb")
    tfidf = joblib.load(tfidf_model)

    length_nor = open("length.pkl", "rb")
    leng = joblib.load(length_nor)

    stopwords_nor = open("stopwords.pkl", "rb")
    stopwords = joblib.load(stopwords_nor)

    capital_nor = open("capitalwords.pkl", "rb")
    capitalwords = joblib.load(capital_nor)

    model = open("model.pkl","rb")
    rf = joblib.load(model)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        feature_length = leng.transform([[len(data[0])]])
        feature_stopwords = stopwords.transform([[spam_classifier.count_stopwords(data[0])]])
        feature_capital = capitalwords.transform([[spam_classifier.count_stopwords(data[0])]])
        feature_tfidf = tfidf.transform(data).toarray()

        vect = pd.DataFrame(feature_tfidf)
        vect.insert(0, 'length', feature_length)
        vect.insert(0, 'stopwords%', feature_stopwords)
        vect.insert(0, 'capital%', feature_capital)
        my_prediction = rf.predict(vect)

    return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)