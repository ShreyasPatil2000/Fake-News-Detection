import re
import joblib
import string
import pandas as pd 
from flask import Flask, render_template, jsonify, request, redirect, url_for
from fakenews import create_bar_chart

app = Flask(__name__,template_folder='templates')
model = joblib.load('model.pkl')
# Global dictionary to store data
data_store = {}

@app.template_filter('content_present')
def content_present(txt):
    return bool(txt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

def wordpre(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

@app.route("/bar_chart")
def bar_chart():
    fig = create_bar_chart() 
    return jsonify(fig.to_json())

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    txt = None
    if request.method == 'POST':
        txt = request.form['txt']
        data_store['txt'] = txt  
        processed_txt = wordpre(txt)  
        txt_series = pd.Series(processed_txt)
        result = model.predict(txt_series)[0]  # Get the first prediction

        data_store['result'] = result  
        return redirect(url_for('result'))
    
    return render_template('index.html', result=None, txt=None)

@app.route('/result')
def result():
    prediction = data_store.get('result', None)  
    txt = data_store.get('txt', None)
    return render_template('index.html', result=prediction, txt=txt)

if(__name__ == '__main__'):
    app.run(debug=True)