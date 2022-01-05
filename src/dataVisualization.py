import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from wordcloud import WordCloud
import plotly.express as px 
import matplotlib.pyplot as plt

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

# Track Utils
from track_utils import add_page_visited_details

# =============Function=============
def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_pickle("datasets/emotion_datasetV2.pkl")
    return data

@st.cache(allow_output_mutation=True)
def load_corpus():
    data = pd.read_pickle("datasets/emotion_corpusV3.pkl")
    return data

@st.cache(allow_output_mutation=True)
def load_month_trend():
    data = pd.read_pickle("datasets/month_trend.pkl")
    return data

@st.cache(persist=True,suppress_st_warning=True)
def get_top_text_ngrams(corpus, ngrams=(1,1), nr=None):
    vec = CountVectorizer(stop_words=stop_words, ngram_range=ngrams).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:nr]


add_stop_words = ['covid', 'long', 'vaccine', 'know', 'people', 'amp', 'time', 'need', 'like', 'year', 'term', 'risk', 'vaccinate', 'symptom','work']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

custom_stop_word_list = ['you know','i mean','yo','dude','couldnt','cant','dont','doesnt','youve',"im",'ive','wasnt','mightnt','hadnt','hvnt','youre','wouldnt','shouldnt','arent','isnt','werent','youll','its','thats', 'covid', 'long', 'vaccine', 'know', 'people', 'amp', 'time', 'need', 'like','year', 'term', 'risk', 'vaccinate', 'symptom', 'work', 'gonna', "gon na", "gon", "na"]
stop_words = stop_words.union(custom_stop_word_list)



def app():

    def title(text,size,color):
        st.markdown(f'<h3 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h3>',unsafe_allow_html=True)

    def header(text):
        st.markdown(f"<p style='color:white;'>{text}</p>",unsafe_allow_html=True)

    st.markdown(f'<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)
    st.markdown("""
        <style>
        blockquote.twitter-tweet {
            display: inline-block;
            font-family: "Helvetica Neue", Roboto, "Segoe UI", Calibri, sans-serif;
            font-size: 12px;
            font-weight: bold;
            line-height: 16px;
            border-color: #eee #ddd #bbb;
            border-radius: 5px;
            border-style: solid;
            border-width: 1px;
            box-shadow: 0 1px 3px rgb(0 0 0 / 20%);
            margin: 10px 5px;
            padding: 8px 16px 16px 16px;
            max-width: 468px;
            transition: transform 500ms ease;
        }
        .twitter-tweet:hover,
        .twitter-tweet:focus-within {
            transform: scale(1.025);
        }
        </style>""",unsafe_allow_html=True)

        
    add_page_visited_details("EDA",datetime.now())

    # loading the data
    df = load_data()
    corpus = load_corpus()
    month_trend = load_month_trend()
    
    st.title("Long Covid on Social Media: Analyzing Twitter Conversations")
    space(1)
    st.markdown("""
    More than 90k real-time tweets on Twitter from May 2021 to September 2021 related to **Long Covid** are analyzed to give up-to-date insights about the post-syndrome of COVID-19 from the lens of social media. What are the keyword trends for long COVID topics across various timeline and different emotions? What is the distribution of emotions towards long COVID topics?
    """)

    # ----------------- Emotion Metrics Percentage -----------------
    title('Distribution of Emotion',40,'black')
    with st.container():
        
        col_1, col_2, col_3 = st.columns((1,1,3))
        with col_1:
            st.metric("Analytical🧐", value = format(len(df[df['emotion']=='analytical'])/len(df)*100,'.2f')+"%")
            st.metric("Neutral😐", format(len(df[df['emotion']=='neutral'])/len(df)*100,'.2f')+"%")
            st.metric("Joy😂", format(len(df[df['emotion']=='joy'])/len(df)*100,'.2f')+"%")
            st.metric("Fear😨😱", format(len(df[df['emotion']=='fear'])/len(df)*100,'.2f')+"%")
        with col_2:
            st.metric("Sadness😔", format(len(df[df['emotion']=='sadness'])/len(df)*100,'.2f')+"%")
            st.metric("Tentative🤔", format(len(df[df['emotion']=='tentative'])/len(df)*100,'.2f')+"%")
            st.metric("Confident😎", format(len(df[df['emotion']=='confident'])/len(df)*100,'.2f')+"%")
            st.metric("Anger😡", format(len(df[df['emotion']=='anger'])/len(df)*100,'.2f')+"%")
        with col_3:
        # ---------------------- Emotion Bar Chart ---------------------
            emotion_count = df['emotion'].value_counts().rename_axis('Emotions').reset_index(name='Counts')
            bar_CC = px.bar(emotion_count, x='Emotions', y='Counts', color='Emotions')
            bar_CC.update_xaxes(tickangle=45)
            bar_CC.update_layout(margin_t=10,margin_b=150)
            st.plotly_chart(bar_CC,use_container_width=True)

    #--------------------------WORD_CLOUD---------------------------
    title('Emotions WordCloud',40,'black')

    unique_emotion = ['analytical','neutral','sadness','joy','anger','tentative','fear','confidence']
    sl = st.slider('Pick Number of Words',50,200)
    
    def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
        return("hsl(240,100%%, %d%%)" % np.random.randint(45,55))
    
    wc = WordCloud(stopwords=stop_words, background_color="white", color_func = grey_color_func, max_font_size=150, random_state=42,max_words=sl, collocations=False)

    plt.rcParams['figure.figsize'] = [40, 40]  #16,6 #40,40
    full_names = unique_emotion

    # Create subplots for each emotion
    for index, emotion in enumerate(corpus.emotion):
        wc.generate(corpus.clean_tweet[emotion])
        
        plt.subplot(4, 2, index+1)  #3,4 #4,2
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(full_names[index], fontsize = 60)
        
    st.pyplot()


    #-------------------------Module 1-----------------------------

    title('Most Popular One Word',40,'black')
    st.caption('removing all the stop words in the sense common words.')

    sl_2 = st.slider('Pick Number of Words',5,50,10, key="1")

    # Unigrams - Most Popular One Keyword
    top_text_bigrams = get_top_text_ngrams(corpus.clean_tweet, ngrams=(1,1), nr=sl_2)
    top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
    x, y = zip(*top_text_bigrams)
    bar_C1 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title='Most Popular One Word', text=y)
    bar_C1.update_traces(textposition="outside", cliponaxis=False)
    bar_C1.update_yaxes(dtick=1, automargin=True)

    st.plotly_chart(bar_C1,use_container_width=True)

    #-------------------------Module 2-----------------------------

    title('Most Popular Two Words',40,'black')

    sl_3 = st.slider('Pick Number of Words',5,50,10, key="2")

    # Unigrams - Most Popular One Keyword
    top_text_bigrams = get_top_text_ngrams(corpus.clean_tweet, ngrams=(2,2), nr=sl_3)
    top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
    x, y = zip(*top_text_bigrams)
    bar_C2 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title='Most Popular Two Word', text=y)
    bar_C2.update_traces(textposition="outside", cliponaxis=False)
    bar_C2.update_yaxes(dtick=1, automargin=True)

    st.plotly_chart(bar_C2,use_container_width=True)

    #-------------------------Module 3-----------------------------

    title('Most Popular Three Words',40,'black')

    header("range")
    sl_4 = st.slider('Pick Number of Words',5,50,10, key="3")

    # Unigrams - Most Popular One Keyword
    top_text_bigrams = get_top_text_ngrams(corpus.clean_tweet, ngrams=(3,3), nr=sl_4)
    top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
    x, y = zip(*top_text_bigrams)
    bar_C3 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title='Most Popular Three Word', text=y)
    bar_C3.update_traces(textposition="outside", cliponaxis=False)
    bar_C3.update_yaxes(dtick=1, automargin=True)

    st.plotly_chart(bar_C3,use_container_width=True)

    #-------------------------Module 4-----------------------------

    title('Top Keywords For Each Month',40,'black')
    months_name = ['May','June','July','August','September']
    months = {'May':'2021-05','June':'2021-06','July':'2021-07','August':'2021-08','September':'2021-09'}

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df_date = df.set_index('Datetime')

    col_1,col_2 =st.columns(2)

    with col_1:
        monthChoice = st.radio("Select Month", ('May','June','July','August','September'))
    with col_2:
        sl_5 = st.slider("Pick Number of Words",5,50,10, key="4")

    # title(f"Top {sl_5} Keywords For {monthChoice}",40,'black')
    # Unigrams - Most Popular One Keyword
    selected_month = months[monthChoice]
    top_text_bigrams = get_top_text_ngrams(df_date.loc[selected_month].clean_tweet, ngrams=(1,1), nr=sl_5)
    top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
    x, y = zip(*top_text_bigrams)
    bar_C4 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title=f'Top {sl_5} Keywords In {monthChoice}', text=y)
    bar_C4.update_traces(textposition="outside", cliponaxis=False)
    # bar_C4.update_layout(title=f'Top KeywordWord In{monthChoice}')
    bar_C4.update_yaxes(dtick=1, automargin=True)

    st.plotly_chart(bar_C4,use_container_width=True)

    #----------------------Line Chart Keywords--------------------------
    title('Top 10 Emerging Words',40,'black')
    line_chart = px.line(month_trend, x='Month', y='Counts', color='Words')
    line_chart.update_traces(mode="markers+lines", hovertemplate=None)
    line_chart.update_layout(hovermode="x unified",plot_bgcolor='aliceblue')
    st.plotly_chart(line_chart,use_container_width=True)


    # -------------------- Emotion selection ------------------------
    
    with st.expander("See Datasets 👇"):
        emotion_list = ['All','analytical','neutral','sadness','joy','anger','tentative','fear','confidence']
        select_emotion = st.selectbox('select emotion',emotion_list)
        # Filtering data
        if select_emotion == 'All':
            df_selected_tweet = df
        else:
            df_selected_tweet = df[(df.emotion.isin([select_emotion]))]

        st.header('Display Tweets of Selected Emotion(s)')
        st.write('Data Dimension: '+str(df_selected_tweet.shape[0]) + ' rows and '+ str(df_selected_tweet.shape[1])+ ' columns.')
        st.dataframe(df_selected_tweet)
