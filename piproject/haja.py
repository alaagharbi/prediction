from  rest_framework.response import Response
from rest_framework.decorators import api_view
import joblib
import os
import sklearn

model_path = os.path.abspath('piproject\linear_regression.pkl')

model = joblib.load(model_path)

@api_view(['POST'])
def getAllusers(request):
    day = request.data["day"]
    year = request.data["year"]
    monthe = request.data["monthe"]
    # Load the trained model file
    # Make a prediction on a new data point
    new_data = [[year, monthe,day]] # replace with your own data
    prediction = model.predict(new_data)
    result = {'prediction': prediction[0]}
    data = {"name": "John", "age": 30,"region":"ariana"}
    return Response(result)

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
@api_view(['POST'])
def analyze_emotions(text):
    # tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    # initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    # analyze sentiment for each sentence and aggregate results
    sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
    for sentence in sentences:
        sentiment = sia.polarity_scores(sentence)
        if sentiment["compound"] > 0.1:
            sentiment_scores["positive"] += 1
        elif sentiment["compound"] < -0.1:
            sentiment_scores["negative"] += 1
        else:
            sentiment_scores["neutral"] += 1
    return sentiment_scores

# example usage
text = "This is a great product! It works perfectly and I would definitely recommend it."
sentiment_scores = analyze_emotions(text)
print(sentiment_scores)
# output: {'positive': 2, 'negative': 0, 'neutral': 0}
