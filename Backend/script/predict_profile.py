from .prepare_text import process_tweets_to_predict

def predict_profiles(tweet, scaler, vectorizer, model, encoder):

    predicted_profile, probability = process_tweets_to_predict(
        tweet = tweet, 
        scaler = scaler, 
        vectorizer = vectorizer, 
        model = model, 
        encoder = encoder)

    if probability > 55:
        prediction = f'I am {probability} confident, {predicted_profile} wrote this Tweet'
    else:
        prediction = f'I am {probability} confident, {predicted_profile} wrote this Tweet. Not sure if this is correct, but what I know is that I still need training!!'

    return {'result': prediction}
