import re
import numpy as np
import string
import spacy
nlp = spacy.load('en_core_web_sm')

def re_remove_url(x):
    return re.sub(r'(http|www.)\S+', '', x).replace('\n', '')

def extract_text_details_to_predict(x):
    uppers = sum([1 for l in x if l.isupper()]) # how many uppercases in each tweet
    punctuations  = sum([1 for l in x if l in string.punctuation]) # how many punctuations in each tweet
    questionsmark = x.count('?') # how many question marks in each tweet
    explainations = x.count('!') # how many explaination marks in each tweet
    return uppers, punctuations, questionsmark, explainations

def cleaning_tweets_to_predict(x):
    try:
        x = str(x)
        tweet = nlp(x)
        tweet = ' '.join([token.lemma_.lower() for token in tweet if not token.is_stop and not token.is_punct and not token.text.isdigit() and len(token.text) > 2])
        return tweet
    except:
        return np.nan
    
def process_tweets_to_predict(tweet, scaler, vectorizer, model, encoder):
    
    tweet = re_remove_url(tweet)
    uppers, punctuations, questionsmark, explainations = extract_text_details_to_predict(tweet)
    tweet = cleaning_tweets_to_predict(tweet)
    
    scaled_featues_to_predict = scaler.transform([[uppers, punctuations, questionsmark, explainations]])
    vectorized_text_to_predict = vectorizer.transform([tweet]).toarray()
    to_predict = scaled_featues_to_predict.tolist()[0] + vectorized_text_to_predict.tolist()[0]
    
    predicted_profile_probability = float(format(round(np.max(model.predict_proba([to_predict])[0]), 3) * 100, '.2f'))
    predicted_profile = model.predict([to_predict])
    predicted_profile = encoder.inverse_transform(predicted_profile)[0]
    
    return predicted_profile, predicted_profile_probability