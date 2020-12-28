import streamlit as st
import requests

# Menu configuration
menu = ["About project", "Give it a try"]
m = st.sidebar.selectbox("", menu)

if m == "About project":
    st.title('Ten Profiles')
    st.write('This project is Multiclass Classification contains 10 different classes of Twitter profiles between politicians and technologest. The accuracy of the deployed model is 76%. The idea of this project is to shawcase how to use streamlit package as frontend for a deployed model as backend and connect both apps via API')
    st.write("GitHub [link](https://github.com/haithamabadla)")
else:
    st.title('Try it now')
    st.write(
        'Place Tweet from one of the following Twitter account and \
        hopefully the ML algorithm will be able to predict who wrote your \
        Tweet. Profiles are: **Barak Obama**, **Donald Trump**, **Bernie Sanders**, \
        **Joe Biden**, **Hillary Clinton**, **Kamala Harris**, **Rudy Giuliani**, \
        **Ivanka Trump**, **Bill Gates**, and **Elon Musk**')
    tweet_text = st.text_area('', max_chars=450, height=150, )
    button = st.button('Predict')
    if button:
        response = requests.get(f"https://fastapi-tweet.herokuapp.com/tweet/{tweet_text}").json()['result']
        st.write('**',response,'**')    

