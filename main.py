import tweepy
from textblob import TextBlob
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import numpy as np
import time

#title
st.title('Tweet Sentiment Analysis')
#markdown
st.markdown('This application is all about tweet sentiment analysis of various twitter users. We can analyse different sentiments of the public using this streamlit app.')
#sidebar
st.sidebar.title('Sentiment analysis of social media data')
# sidebar markdown
st.sidebar.markdown("🛫We can analyse public review from this application.🛫")

# Keys
api_key = "3pVTC13H7GCChyeNR4Nmxm0jh"
api_key_secret = "q1dVMhsBVUqK092beW1evmwud9JCmlBuFDxrq2YOGiBSJxC0Go"
access_token = "848775572473991169-DA1WgxLVcVRVOVo3g4wm6yDMPX6h9xy"
access_token_secret = "loyzMnFlVPEgnQE3PEMFoyBdKbjVGRF38aKOctJPkadsK"

# Authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
with st.form(key = "form1"):
    keyword = st.text_input(label= "Enter the Keyword")
    submit = st.form_submit_button(label="Analyze this keyword")

limit=200
columns = ['User', 'Tweet']
data = []
time.sleep(15)
tweets = tweepy.Cursor(api.search_tweets, q=keyword, count=100, tweet_mode='extended', lang='en').items(limit)
for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text])
df = pd.DataFrame(data, columns=columns)
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) #Removed @mentions
    text = re.sub(r'#', '', text) #Removed the '#' symbol
    text = re.sub(r':', '', text) #Removed the colons
    text = re.sub(r'RT[\s]+', '', text) #Removed the Re-Tweets
    text = re.sub(r'https?:\/\S+', '', text) # remove the hyperlinks
    text = re.sub(r'https', '', text)
    return text

df['Tweet'] = df['Tweet'].apply(cleanTxt)
# print(df['Tweet'])
data = df['Tweet']
#checkbox to show data
if st.checkbox("Show Data"):
    st.write(data.head(50))
#subheader
st.sidebar.subheader('Tweets Analyser')
# radio buttons
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df['Subjectivity'] = df['Tweet'].apply(getSubjectivity)
df['Polarity'] = df['Tweet'].apply(getPolarity)
allWords = ''.join([twts for twts in df['Tweet']])
wordCloud = WordCloud().generate(allWords)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
# plt.show()
# Create a function to compute the negative, neutral and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)
data = df
print(data)
time.sleep(15)
tweets=st.sidebar.radio('Sentiment Type',('Positive','Negative','Neutral'))
st.write(data.query('Analysis==@tweets')[['Tweet']].sample(1).iat[0,0])
st.write(data.query('Analysis==@tweets')[['Tweet']].sample(1).iat[0,0])
st.write(data.query('Analysis==@tweets')[['Tweet']].sample(1).iat[0,0])
# selectbox + visualisation
# An optional string to use as the unique key for the widget. If this is omitted, a key will be generated for the widget based on its content.
# Multiple widgets of the same type may not share the same key.
select=st.sidebar.selectbox('Visualisation Of Tweets',['Histogram','Pie Chart'],key=1)
sentiment=data['Analysis'].value_counts()
sentiment=pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})
st.markdown("###  Sentiment count")
if select == "Histogram":
    fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
    st.plotly_chart(fig)
else:
    fig = px.pie(sentiment, values='Tweets', names='Sentiment')
    st.plotly_chart(fig)


plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Red')
    
plt.title('Sentimental Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
st.pyplot()

