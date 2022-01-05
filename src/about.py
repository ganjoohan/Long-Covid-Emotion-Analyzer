import streamlit as st
from track_utils import add_page_visited_details
from datetime import datetime


def app():

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
        
    #st.subheader("About")
    add_page_visited_details("About",datetime.now())
    st.title("About the Application")

    st.markdown("""

    <h2 style="font-weight:bolder;font-size:20px;color:#216fdb;text-align:left;">What is this App about?</h2>
    
    This project aims to build a long COVID emotion analyzer. The application can analyze how people react to the long COVID and what are their emotions. The application can find out the trends of keywords used by people regarding the long COVID issues. The application can also analyze the emotion from the user input text. 

    
    <h2 style="font-weight:bolder;font-size:20px;color:#216fdb;text-align:left;">Who is this App for?</h2>

    <p>Anyone can use this App completely for free! The target users are the people who are concerned about long COVID issues and long COVID patients. If you are interest in finding out how people feel about the pandemic and long COVID issues, this application would be a good choice for you to explore.
    
    If you like it ❤️, show your support by sharing 👍</p>
    
    <h2 style="font-weight:bolder;font-size:20px;color:#216fdb;text-align:left;">What is the features of this App?</h2>

    + Data Exploration and Analysis
    + Emotion Prediction
    + Monitoring Application


    <h2 style="font-weight:bolder;font-size:20px;color:#216fdb;text-align:left;">What is the objectives of this App?</h2>

    1. To explore the trend of keywords across the various timeline and different emotions
        - What are the keyword trends for long COVID topics?
            

    2. To determine the emotions using supervised algorithms
        - What machine learning algorithms can be used to determine the emotions from texts?   
        
        
    3. To assess the effectiveness of the emotion model by using evaluation metrics
        - What are metrics can be used to evaluate the emotion model?   


    4. To develop a product for the emotion analyzer
        - How to share the insights of the data and the application of the emotion model to stakeholders?   


    <h2 style="font-weight:bolder;font-size:20px;color:#216fdb;text-align:left;">Data Information</h2>

    The dataset to be used in this project is scraped from Twitter using Twitter API.     
    To access the dataset:   
    https://drive.google.com/drive/folders/1cT9FzTWmjATdy7rlkrcRj_NU2zKCtYZE?usp=sharing

    <h2 style="font-weight:bolder;font-size:20px;color:#216fdb;text-align:left;">Source Code</h2>

    https://github.com/ganjoohan/Long-Covid-Emotion-Analyzer

    <h2 style="font-weight:bolder;font-size:20px;color:#216fdb;text-align:left;">About me</h2>
    I'm originally from Malacca, Malaysia, currenlty a third-year Bachelor of CS Data Science student at the University of Malaya!   

    <br />
    <br />

    ###### Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with ❤️ by [@JooHan](https://joohan.soho68.com/) &nbsp | &nbsp [![GitHub followers](https://img.shields.io/github/followers/ganjoohan?label=Github&style=social)](https://github.com/ganjoohan) &nbsp | &nbsp [![this is an image link](https://camo.githubusercontent.com/d7b9f7e3f8af9348678c5042440844da48b892fb320482f313d28366d10c25d5/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c696e6b6564496e2d2532333030373742352e7376673f267374796c653d666f722d7468652d6261646765266c6f676f3d6c696e6b6564696e266c6f676f436f6c6f723d7768697465)](https://www.linkedin.com/in/gan-j-919226136/)




    """, unsafe_allow_html=True)
