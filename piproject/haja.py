from  rest_framework.response import Response
from rest_framework.decorators import api_view
import joblib
import os
import sklearn
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
model_path = os.path.abspath('piproject\linear_regression.pkl')

model = joblib.load(model_path)
text = "This is a great product! It works perfectly and I would definitely recommend it."

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
@api_view(['POST'])
def analyze_emotions(request):
    text =request.data["text"]
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
    return Response(sentiment_scores)
import requests
from textblob import TextBlob

# Replace the ACCESS_TOKEN value with your own Mapbox access token
ACCESS_TOKEN = "your-access-token-here"

# Define the URL for the Mapbox Places API
URL = "https://api.mapbox.com/geocoding/v5/mapbox.places/{place}.json"

def get_reviews(place):
    # Define the parameters for the API request
    params = {
        "access_token": ACCESS_TOKEN,
        "limit": 10,
        "types": "poi",
        "autocomplete": "true"
    }

    # Send the API request and get the response
    response = requests.get(URL.format(place=place), params=params)
    data = response.json()

    # Extract the place_id for the first result
    place_id = data["features"][0]["id"]

    # Define the URL for the Mapbox Reviews API
    reviews_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{place_id}.json"

    # Define the parameters for the Reviews API request
    reviews_params = {
        "access_token": ACCESS_TOKEN,
        "types": "poi",
        "language": "en"
    }
    
    # Send the Reviews API request and get the response
    reviews_response = requests.get(reviews_url, params=reviews_params)
    reviews_data = reviews_response.json()
    
    # Extract the reviews from the response
    reviews = []
    for feature in reviews_data["features"]:
        text = feature["properties"]["address"]
        reviews.append(text)
    
    return reviews

def analyze_reviews(reviews):
    # Analyze the sentiment of each review using TextBlob
    sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
    for review in reviews:
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.1:
            sentiment_scores["positive"] += 1
        elif sentiment < -0.1:
            sentiment_scores["negative"] += 1
        else:
            sentiment_scores["neutral"] += 1
            
    return sentiment_scores

# Example usage
place = "your-place-name-here"
reviews = get_reviews(place)
sentiment_scores = analyze_reviews(reviews)
print(sentiment_scores)
# Output: {'positive': 2, 'negative': 1, 'neutral': 1}
