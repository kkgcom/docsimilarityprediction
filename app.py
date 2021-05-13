import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    doc1=request.form.get('doc1')
    doc2=request.form.get('doc2')
    doc = [doc1 , doc2]

    cv = CountVectorizer(stop_words='english')
    matrix = cv.fit_transform(doc)
    
    df = pd.DataFrame(matrix.toarray() , columns =cv.get_feature_names() , index = ['doc1' , 'doc2'] )
    prediction=cosine_similarity(doc1.reshape(1,-1),doc2.reshape(1,-1))
    if round(prediction[0],2)==1.0:
        output = 'no similarity between the given 2 docs'
    else:
        output = 'similarity between the given 2 docs'
    return render_template('index.html',prediction_text='There is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)