import streamlit as st
from track_utils import add_prediction_details
from datetime import datetime
import pandas as pd
import numpy as np
import neattext as nt
from neattext.functions import clean_text
import joblib
from sklearn.feature_extraction import text 
# from PIL import Image
# import altair as alt

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

# Function
pipe_lr = joblib.load(open("models/emotion_classifier_lr_model_22_Dec_2021.pkl","rb"))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# def get_wordnet_pos(treebank_tag):
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return None

# lemmatizer = WordNetLemmatizer()

# def clean_text_round2(text):
#     tokens = nltk.word_tokenize(text)
#     tagged = nltk.pos_tag(tokens)
#     full_text = ''
#     for word, tag in tagged:
#         wntag = get_wordnet_pos(tag)
#         if wntag is None:
#             lemma = lemmatizer.lemmatize(word)
#         else:
#             lemma = lemmatizer.lemmatize(word, pos=wntag)
#         full_text += lemma + ' '
#     return full_text

add_stop_words = ['covid', 'long', 'vaccine', 'know', 'people', 'amp', 'time', 'need', 'like', 'year', 'term', 'risk', 'vaccinate', 'symptom','work']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

custom_stop_word_list = ['you know','i mean','yo','dude','couldnt','cant','dont','doesnt','youve',"im",'ive','wasnt','mightnt','hadnt','hvnt','youre','wouldnt','shouldnt','arent','isnt','werent','youll','its','thats', 'covid', 'long', 'vaccine', 'know', 'people', 'amp', 'time', 'need', 'like','year', 'term', 'risk', 'vaccinate', 'symptom', 'work', 'gonna', "gon na", "gon", "na"]
stop_words = stop_words.union(custom_stop_word_list)

def cleantext(docx):
    docxFrame = nt.TextFrame(text=docx)
    docxFrame.remove_hashtags()
    docxFrame.remove_userhandles()
    docxFrame.remove_multiple_spaces()
    docxFrame.remove_urls()
    docxFrame.remove_emails()
    docxFrame.remove_numbers()
    docxFrame.remove_emojis()
    docxFrame.remove_puncts()
    docxFrame.remove_special_characters()
    docxFrame.remove_non_ascii()
    docxFrame.remove_stopwords()
    
    cleanDocx = docxFrame.text
    cleanDocx = clean_text(cleanDocx, contractions=True, stopwords=True)
    cleanDocx = ' '.join(term for term in cleanDocx.split() if term not in stop_words)
    # cleanDocx = clean_text_round2(cleanDocx)
    return cleanDocx

emotions_emoji_dict = {"analytical":"🧐", "sadness":"😔", "neutral":"😐","tentative":"🤔","joy":"😂","confident":"😎","fear":"😨😱","anger":"😡"}


# Disable Some Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)


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
            margin: 10px 15%;
            padding: 8px 16px 16px 16px;
            max-width: 468px;
            transition: transform 500ms ease;
            

        }
        .twitter-tweet:hover,
        .twitter-tweet:focus-within {
            transform: scale(1.025);
        }
        </style>""",unsafe_allow_html=True)

    # add_page_visited_details("Home",datetime.now())
    # Page title
    #st.title("Long Covid Emotion Analyzer")
    # col1, col2, col3, col4 = st.columns([1,3,3,1])
    # img = Image.open("images/logo.jpg")
    # with col1:
    #     st.write("")

    # with col2:
    #     st.image(img, use_column_width=True)

    # with col3:
    #     space(2)
    #     st.markdown("""
    #     ## This application analyze the emotions of people about long COVID topic on Twitter and predict the emotions!
    #     """)
    #     space(1)
    #     st.markdown("""##### People show various emotions in daily communications. This emotion predictor analyses emotions in what people write online such as tweets. It will predict whether they are joy, fear, sadness, anger, analytical, confident and tentative.""")
    # with col4:
    #     st.write("")
    
    st.markdown('<h1 style="font-weight:10;font-size: 50px;font-family:Source Sans Pro, sans-serif;text-align:center;">Long Covid Emotion Analyzer</h1>',unsafe_allow_html=True)
    space(2)
    # Long Covid Emotion Analyzer
    # space(1)
    #st.write("***")
    col_1, col_2, col_3 = st.columns([1,8,1])

    with col_1:
        st.write()
    
    with col_2:
        st.subheader("Emotion Analyzer In Text")
        space(1)
        st.markdown("**Instructions:** Type in your text")

        with st.form(key='emotion_form'):
            raw_text = st.text_area('Type Here',"Long Covid brings lots of negative and bad effect to the patient. I feel sorry to those who are suffering from Long Covid symptoms")
            cleanDocx = cleantext(raw_text)
            submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        #st.balloons()  #display some balloons effect xD
        col1, col2, col3, col4 = st.columns([1,2,4,1])
        # col1,col2 = st.columns(2)

        # Apply Prediction Funtion Here
        prediction = predict_emotions(cleanDocx)
        probability = get_prediction_proba(cleanDocx)

        add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

        with col2:
            # st.success("Original Text")
            # st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction,emoji_icon))
            st.write("Score:{:.0%}".format(np.max(probability)))
        
        with col3:
            # st.success("Preprocessing Text")
            # st.write(cleanDocx)

            st.success("Emotion Score")
            #st.write(probability)
            proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            porba_df_clean = proba_df.T.reset_index()
            porba_df_clean.columns = ["emotions","probability"]

            # fig = alt.Chart(porba_df_clean,height=400).mark_bar().encode(x='emotions',y='probability', color='emotions')
            # st.altair_chart(fig, use_container_width=True)
            # ---------------------- Emotion Bar Chart ---------------------
            import plotly.express as px 
            bar_CC = px.bar(porba_df_clean, x='emotions', y='probability', color='emotions',color_discrete_sequence=px.colors.qualitative.T10)
            # https://plotly.com/python/discrete-color/

            bar_CC.update_xaxes() #tickangle=0
            bar_CC.update_layout() #margin_t=10,margin_b=150
            st.plotly_chart(bar_CC,use_container_width=True)
    else:
        with col_2: 
            st.write("*Analysis of text will appear here after you click the 'Analyze' button*")
    
