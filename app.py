import re
import joblib
import string
import pandas as pd 
import os
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_mail import Mail, Message
from dotenv import load_dotenv

app = Flask(__name__,template_folder='templates')
model = joblib.load('model.pkl')
# Global dictionary to store data
data_store = {}

load_dotenv()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')   
app.config['MAIL_USE_TLS'] = True 
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

@app.template_filter('content_present')
def content_present(txt):
    return bool(txt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        msg = Message(subject, sender=os.getenv('MAIL_USERNAME'), recipients=[os.getenv('MAIL_USERNAME')])
        msg.body = f'''From: {name} <{email}>
        Subject: {subject}
        Message:
        {message}'''
        try:
            mail.send(msg)
            flash('Your message has been sent successfully!', 'success')
        except Exception as e:
            flash('An error occurred while sending your message. Please try again.', 'error')
        
        return redirect(url_for('contact'))
    
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