import joblib
from script.predict_profile import predict_profiles
from fastapi import FastAPI

app = FastAPI()

profiles_encoder = joblib.load('ml_models/profiles_encoder.sav')
extra_features_scaler = joblib.load('ml_models/extra_features_scaler.sav')
tfidf_vectorizer = joblib.load('ml_models/tfidf_vectorizer.sav')
logreg_model = joblib.load('ml_models/logreg_model.sav')

@app.get('/tweet/{tweet}')
def tweet(tweet: str):
    return predict_profiles(
        tweet = tweet, 
        scaler = extra_features_scaler, 
        vectorizer = tfidf_vectorizer, 
        model = logreg_model, 
        encoder = profiles_encoder)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')