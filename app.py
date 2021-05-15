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
    prediction=cosine_similarity(matrix[0].reshape(1,-1),matrix[1].reshape(1,-1))
    output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text='The similarity between the docs are: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)