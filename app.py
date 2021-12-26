# Core Pkgs
import requests
import streamlit as st
import streamlit.components.v1 as components
import json
from streamlit_lottie import st_lottie

# Visualization
from wordcloud import WordCloud
import altair as alt
import plotly.express as px 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image

# EDA Pkgs
import pandas as pd
import numpy as np
from neattext.functions import clean_text
from datetime import datetime

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer

# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

# Utils
import joblib

# Function
pipe_lr = joblib.load(open("models/emotion_classifier_lr_model_22_Dec_2021.pkl","rb"))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

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

def title(text,size,color):
    st.markdown(f'<h1 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h1>',unsafe_allow_html=True)

def header(text):
    st.markdown(f"<p style='color:white;'>{text}</p>",unsafe_allow_html=True)

def twitter(text):
    st.markdown(f"<p style='overflow-x: scroll'>{text}</p>",unsafe_allow_html=True)

@st.cache(persist=True,suppress_st_warning=True)
def get_top_text_ngrams(corpus, ngrams=(1,1), nr=None):
    vec = CountVectorizer(ngram_range=ngrams).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:nr]

# def cleantext(docx):
#     cleanDocx = clean_text(docx)
#     return cleanDocx

emotions_emoji_dict = {"analytical":"🧐", "sadness":"😔", "neutral":"😐","tentative":"🤔","joy":"😂","confident":"😎","fear":"😨😱","anger":"😡"}

# Disable Some Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

img = Image.open("images/webicon.jpg")

st.set_page_config(
     page_title="Long Covid Emotion Analyzer",
     page_icon= img,
     layout="centered",
     initial_sidebar_state="auto",
     menu_items={
         'About': "### MEsut_GAn"
     }
 )

st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)

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
    }
    </style>""",unsafe_allow_html=True)

def main():
    
    # loading the data
    df = load_data()
    corpus = load_corpus()
    month_trend = load_month_trend()
    
    menu = ["Home","Emotion Predictor","EDA","Monitor","Documentation","About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home",datetime.now())
        # Page title
        #st.title("Long Covid Emotion Analyzer")
        img = Image.open("images/logo.jpg")
        st.image(img, use_column_width=True)

        space(1)
        st.write("""
        # Long Covid Emotion Analyzer
        
        This app analyze the emotions of people about long COVID topic on Twitter and predict the emotions!

        ***
        """)
        
        #Audio
        # audio_file = open("video/backgroundmusic.mp3","rb").read()
        # st.audio(audio_file,format='audio/mp3')

        home_col_1, home_col_2 = st.columns((2,1))
        with home_col_1:
            st.subheader("What is Long Covid?")
            st.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")

        with home_col_2:
            space(1)
            lottie_coding = load_lottiefile("video/covid19.json")
            st_lottie(
                lottie_coding,
                speed=2,
                reverse=False,
                loop=True,
                quality="low", #medium / high
                height=None,
                width=None,
                key=None,
            )

        # Video
        st.write("***")
        space(2)
        st.markdown("Let's watch a video about Long-COVID recovery 🎥")
        st.video("https://youtu.be/GrCKc3X2-1Y")

        # Sample tweets for Each Emotions
        space(1)
        st.subheader("Sample Tweets For Each Emotions Categories")
        with st.container():
            
            col_11, col_22 = st.columns(2)

            with col_11:

                # =============== Analytical ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Analytical🧐</h4>",unsafe_allow_html=True)
                st.caption("A person's reasoning and analytical attitude about things. An analytical person might be perceived as intellectual, rational, systematic, emotionless, or impersonal.")
                st.markdown(""" 
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Going forward, we will:<br><br>- establish sensible mandatory vaccine rules in federal jurisdiction<br>- support provinces that implement vaccine credentials<br>- procure boosters for those who need them<br>- invest in research to study COVID&#39;s long-term health impacts</p>&mdash; Nate Erskine-Smith (@beynate) <a href="https://twitter.com/beynate/status/1432020468660412423?ref_src=twsrc%5Etfw">August 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">🧵 Even in other diseases, it is essential to find doctors determined to listen to the patients to reach diagnosis through a correct assessment of body disfunction using right imaging technique<br><br>In <a href="https://twitter.com/hashtag/LongCovid?src=hash&amp;ref_src=twsrc%5Etfw">#LongCovid</a> <br>1.PET-SCAN <br>2.Heart RMI<br>2.SPECT-CT scan <br><br>gave the first evidence 👇 <a href="https://t.co/38nW90Egr8">https://t.co/38nW90Egr8</a></p>&mdash; Long Covid Italia (@LongCovidItalia) <a href="https://twitter.com/LongCovidItalia/status/1399114470685872128?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Medical experts don&#39;t know how long immunity lasts after getting COVID-19. Vaccination will effectively reduce the risk of severe illness and hospitalization from COVID-19.  Find a vax site near you, visit: <a href="https://t.co/v6XjjE8u28">https://t.co/v6XjjE8u28</a> <a href="https://twitter.com/hashtag/VaxUpChatham?src=hash&amp;ref_src=twsrc%5Etfw">#VaxUpChatham</a> <a href="https://twitter.com/hashtag/VaxUpSavannah?src=hash&amp;ref_src=twsrc%5Etfw">#VaxUpSavannah</a> <a href="https://twitter.com/hashtag/SavannahStrong?src=hash&amp;ref_src=twsrc%5Etfw">#SavannahStrong</a> <a href="https://t.co/Uqf5mcciHq">pic.twitter.com/Uqf5mcciHq</a></p>&mdash; City of Savannah (@cityofsavannah) <a href="https://twitter.com/cityofsavannah/status/1399151644101812226?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">to clarify: it wasn&#39;t COVID fright...I&#39;m vaccinated as was anyone I had real long prolonged contact with. More just the overwhelmingness of it all. It&#39;s a lot to take in all at once.</p>&mdash; Dr. Jacob A. Cohen (@MusicoloJake) <a href="https://twitter.com/MusicoloJake/status/1399148829736308742?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">The long haul effects of COVID are what kept me and my family at home until vaccinated. If you care about yourself, your family, and you want to lead a healthy life - don’t wait. <a href="https://twitter.com/hashtag/takeyourshot?src=hash&amp;ref_src=twsrc%5Etfw">#takeyourshot</a> <a href="https://twitter.com/hashtag/CovidVaccine?src=hash&amp;ref_src=twsrc%5Etfw">#CovidVaccine</a> <a href="https://t.co/pAptI4EReh">https://t.co/pAptI4EReh</a></p>&mdash; JAPhelan (@jphelan713) <a href="https://twitter.com/jphelan713/status/1398620491821223937?ref_src=twsrc%5Etfw">May 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Genuine question. Anyone know how to get rid of post Covid fatigue? Or how long it takes to go away? It’s been a month and everyday I’m tired as if I’ve run a marathon. Never used to happen before.</p>&mdash; Karan Singh (@karansinghmagic) <a href="https://twitter.com/karansinghmagic/status/1397842364287311873?ref_src=twsrc%5Etfw">May 27, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                </div>
                """,unsafe_allow_html=True)
                
                # =============== Joy ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Joy😂</h4>",unsafe_allow_html=True)
                st.caption("Joy (or happiness) has shades of enjoyment, satisfaction, and pleasure. Joy brings a sense of well-being, inner peace, love, safety, and contentment.")
                st.markdown(""" 
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">It’s been a long road, and we are all looking forward to the end of the COVID-19 outbreak. While we’re not out of the woods yet, it’s a good time to look back at what we have learned from this experience. <a href="https://t.co/8NTOuiwvqn">https://t.co/8NTOuiwvqn</a> <a href="https://t.co/RNX4Hil7U2">pic.twitter.com/RNX4Hil7U2</a></p>&mdash; Newport Hospital (@NewportHospital) <a href="https://twitter.com/NewportHospital/status/1442595551820849152?ref_src=twsrc%5Etfw">September 27, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I’m glad I was alive for the COVID-19 era. It illuminated me on a fraction of society I long looked down on as silly,superstitious &amp; folly.A part of society that is willing to hold untrustworthy institutions accountable regardless of the consequences. Liberation struggle vibes.</p>&mdash; Nafimane Halweendo. (@Naffy101) <a href="https://twitter.com/Naffy101/status/1432063616967118848?ref_src=twsrc%5Etfw">August 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I’M GOING TO WATCH FREE GUY !!!!! seeing chan’s bestie on the big screen 😁😁 haven’t been to the cinema in soooo long cuz of covid so yay!!!</p>&mdash; ➳❥ lyds (@inniessmile) <a href="https://twitter.com/inniessmile/status/1432038879817838596?ref_src=twsrc%5Etfw">August 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">4 Months Home ❤️ after getting COVID in January ❤️Thankyou from the bottom of our Hearts to all the NHS Staff in Derriford ❤️Not once but twice in ICU 😩but we will get there now with this Long COVID ! Jabs not available then but are now get your Jabs in 😘 <a href="https://t.co/lCOp5mkSGo">pic.twitter.com/lCOp5mkSGo</a></p>&mdash; kerri (@kerrimason76) <a href="https://twitter.com/kerrimason76/status/1420444709735047175?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">🙋🏻‍♀️ still excited about it and my long term COVID symptoms are gone 👍</p>&mdash; THEE Daniela_Brooklyn (@DanielaBrookly1) <a href="https://twitter.com/DanielaBrookly1/status/1408802295400177672?ref_src=twsrc%5Etfw">June 26, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">For the first time in 15 months my vaccinated ass went to a restaurant and a movie. It was really fun but now a need a week long nap because that was a lot of stimulation after living in a covid Skinner box. (Quiet Place Too is awesome)</p>&mdash; Covid, turning Florida blue one corpse at a time (@megfug) <a href="https://twitter.com/megfug/status/1399146776968318978?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                </div>
                """,unsafe_allow_html=True)
            
            with col_22:

                # =============== Sadness ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Sadness😔</h4>",unsafe_allow_html=True)
                st.caption("Sadness indicates a feeling of loss and disadvantage. When a person is quiet, less energetic, and withdrawn, it can be inferred that they feel sadness.")
                st.markdown(""" 
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">My niece caught covid on her 1st day back and went back to school after isolation, really struggled pushing herself to catch up. She&#39;s progressively getting worse with fatigue, insomnia...heartbreaking as I&#39;m at 18 months with long covid 🙏</p>&mdash; carol anne manson be (@carolannemanso2) <a href="https://twitter.com/carolannemanso2/status/1441472688074203137?ref_src=twsrc%5Etfw">September 24, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Day 489 the burning pain in the soles of my feet is back with a vengeance 😪 my stomach continues to feel really bloated yet I have never been more regular. <a href="https://twitter.com/hashtag/FBLC?src=hash&amp;ref_src=twsrc%5Etfw">#FBLC</a> <a href="https://twitter.com/hashtag/CountLongCovid?src=hash&amp;ref_src=twsrc%5Etfw">#CountLongCovid</a> <a href="https://twitter.com/hashtag/longhauler?src=hash&amp;ref_src=twsrc%5Etfw">#longhauler</a> <a href="https://twitter.com/hashtag/LongCovid?src=hash&amp;ref_src=twsrc%5Etfw">#LongCovid</a></p>&mdash; Jacqueline H #FBLC (@jackpack1968) <a href="https://twitter.com/jackpack1968/status/1432402885044850688?ref_src=twsrc%5Etfw">August 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I feel so sad for his family and I hope his death - and that of other radio hosts like him - isn&#39;t in vain, and saves people from death or long covid and hospital bills by convincing them to get the vaccine.</p>&mdash; ginny_c (@ginny_bear1) <a href="https://twitter.com/ginny_bear1/status/1432342490598739971?ref_src=twsrc%5Etfw">August 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I feel so sad for that baby 💔 I hope the baby doesn&#39;t have long covid issues</p>&mdash; Andrea 💛💙 (@EfCovid19) <a href="https://twitter.com/EfCovid19/status/1431823201596854273?ref_src=twsrc%5Etfw">August 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Covid long hauler here. Tonight I was on a surprise birthday call to my college best friend and I got stuck on the word “unexpected”. My brain fog made it temporarily impossible for me to say. I got too embarrassed and left the call. Brain fog is real and awful.</p>&mdash; Feminist_FR (@covid_longhaul) <a href="https://twitter.com/covid_longhaul/status/1420599180859424768?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">19 percent of 37 people out of 1500 health workers - so 7 people out of 1500 have at least one lingering symptom after six weeks. Is that a cough that will pass? Is that long covid? I don’t know.</p>&mdash; Jon Lovett (@jonlovett) <a href="https://twitter.com/jonlovett/status/1420540119090569218?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                </div>
                """,unsafe_allow_html=True)

                # =============== Neutral ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Neutral😐</h4>",unsafe_allow_html=True)
                st.caption("Feeling indifferent, nothing in particular, and a lack of preference one way or the other")
                st.markdown(""" 
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sept. 25, 2021 Northwest Suburban College virtual conference on Covid-19: Where are we now.<br>Multiple speakers at this conference including Leonard Jason. Covid-19 symptoms among Long Haulers and those with ME/CFS. <a href="https://t.co/Dc0eADpBhI">https://t.co/Dc0eADpBhI</a></p>&mdash; Leonard Jason (@CenterRes) <a href="https://twitter.com/CenterRes/status/1442086251205586955?ref_src=twsrc%5Etfw">September 26, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">There are already tens of thousands of children with long covid in the UK; it will take months to see how many from the current, exponentially increasing surge are joining them.<br><br>When people say that covid is becoming a disease of the unvaccinated, that includes children. <a href="https://t.co/jWYQu8Oqrs">https://t.co/jWYQu8Oqrs</a></p>&mdash; Rachel Thomas (@math_rachel) <a href="https://twitter.com/math_rachel/status/1432162787753529351?ref_src=twsrc%5Etfw">August 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Oxford West and Abingdon MP Layla Moran says a long covid compensation scheme is &#39;urgently needed&#39; for key workers.<br><br>At an event on Friday, she heard from nurses who said some colleagues living with the condition will take &#39;years&#39; to recover.<a href="https://twitter.com/hashtag/HeartNews?src=hash&amp;ref_src=twsrc%5Etfw">#HeartNews</a> <a href="https://t.co/d3IIivTNjj">pic.twitter.com/d3IIivTNjj</a></p>&mdash; Thames Valley News (@HeartThamesNews) <a href="https://twitter.com/HeartThamesNews/status/1409413708573511680?ref_src=twsrc%5Etfw">June 28, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">The emergence of Long COVID provides an opportunity to implement new approaches to integrated, well coordinated, multidisciplinary, person centred care. Read more in our latest issues brief: <a href="https://t.co/Jio51QE3kH">https://t.co/Jio51QE3kH</a></p>&mdash; The Deeble Institute for Health Policy Research (@DeebleInstitute) <a href="https://twitter.com/DeebleInstitute/status/1397379491211714560?ref_src=twsrc%5Etfw">May 26, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Nepal is battling a surge in Covid-19 cases and desperately needs help to critical medical supplies, equipment and oxygen. Singapore has long benefited from the service of Nepali Gurkhas, it&#39;s now time to help them in their hour of need. Here&#39;s how you can help.</p>&mdash; Sulaiman Daud (@thesulaimandaud) <a href="https://twitter.com/thesulaimandaud/status/1397417230648766467?ref_src=twsrc%5Etfw">May 26, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Two new studies show that people who have recovered from COVID-19 could have long-lasting immunity even after antibodies fade. Now, Houston based infectious disease experts are weighing in on whether or not booster shots are needed.<br> <a href="https://t.co/CAtNbuUfjT">https://t.co/CAtNbuUfjT</a></p>&mdash; HOUmanitarian (@HOUmanitarian) <a href="https://twitter.com/HOUmanitarian/status/1399125220787736577?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                </div>
                """,unsafe_allow_html=True)

            col_33, col_44 = st.columns(2)

            with col_33:
                # =============== Tentative ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Tentative🤔</h4>",unsafe_allow_html=True)
                st.caption("A person's degree of inhibition. Feel hesitant or unsure about something. Might be perceived as questionable, doubtful, or debatable.")
                st.markdown(""" 
                
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I wonder if a person dies of long covid are they added to the reported &#39;covid deaths&#39;? I guess, ironically, if they are then they would be classed as having an underlying condition - long covid.</p>&mdash; 🖤💛❤️ Lynne Murphy (@lynnemurphy1) <a href="https://twitter.com/lynnemurphy1/status/1442321514724925444?ref_src=twsrc%5Etfw">September 27, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I’m currently in the same situation as you and I called up the doctors and he gave me a prescription of antibiotics and an inhaler because he thinks it’s a possible chest infection due to long covid! Maybe give docs a ring and see if they can give you anything to help💚</p>&mdash; ⱠØⱠ₳ ₮ⱧØⱤ₦Ɇ 🔪 (@mulletbitchxo) <a href="https://twitter.com/mulletbitchxo/status/1432484126372900869?ref_src=twsrc%5Etfw">August 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Reacting to a rise in cases, sans rising deaths or hospitalizations, does imply a zero-covid end goal. <br><br>I believe this is where a counter argument would begin, “but long covid…” <br><br>The hope for a speedy eradication is so strong. Many are working to keep that hope alive.</p>&mdash; Jake Sheff (@Jake_Sheff) <a href="https://twitter.com/Jake_Sheff/status/1420769048359407621?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Nyambi Nyambi is great as Jay Dipersia and I haven&#39;t gushed enough about him. Can&#39;t believe Jay has long covid though, that&#39;s gonna be difficult for him</p>&mdash; Ruxandra Grrrr 👾 (@trifoi) <a href="https://twitter.com/trifoi/status/1408508690194472962?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">These people are ignorant and nasty. Don&#39;t let them get to you. I hope your long covid improves</p>&mdash; Dr.ajEdenvalley🐀🌺🌺🌺 (@ajEdenvalley1) <a href="https://twitter.com/ajEdenvalley1/status/1399111480482488320?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Friends, see this new protocol for long Covid.<br><br>There is hope!<a href="https://t.co/RKxJFCPiJY">https://t.co/RKxJFCPiJY</a> <a href="https://t.co/RzvCZMGOQi">https://t.co/RzvCZMGOQi</a></p>&mdash; Linda Richmond (@comments_007) <a href="https://twitter.com/comments_007/status/1399131727298142210?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

                </div>
                """,unsafe_allow_html=True)
                
                # =============== Confident ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Confident😎</h4>",unsafe_allow_html=True)
                st.caption("A confident tone indicates a person's degree of certainty. A confident person might be perceived as assured, collected, hopeful, or egotistical.")
                st.markdown(""" 
                
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">What a phenomenal thread calling out some absolute superstars (including <a href="https://twitter.com/itsbodypolitic?ref_src=twsrc%5Etfw">@itsbodypolitic</a>, itself ofc!). Thank you all for what you do tirelessly and selflessly for <a href="https://twitter.com/hashtag/LongCovid?src=hash&amp;ref_src=twsrc%5Etfw">#LongCovid</a> 🙏🏻🙏🏻🤗 <a href="https://t.co/nqva0IZdIt">https://t.co/nqva0IZdIt</a></p>&mdash; Putrino Lab (@PutrinoLab) <a href="https://twitter.com/PutrinoLab/status/1442276768652595200?ref_src=twsrc%5Etfw">September 26, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">So the wean and i(and my cousin) climbed Croagh Patrick today 🇮🇪⛪<br>Took us 7 hours up and down... I slowed them right down and long COVID killed my left lung completely but we did it 🇮🇪💚🍀 <a href="https://t.co/whexyE6uwO">pic.twitter.com/whexyE6uwO</a></p>&mdash; Eileen McGovern (@lene2104) <a href="https://twitter.com/lene2104/status/1421214511617413122?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Absolutely excellent. Packed to the brim full of information. A long watch but worth every second.<br>👏👏 thank you to <a href="https://twitter.com/ukcolumn?ref_src=twsrc%5Etfw">@ukcolumn</a> for bringing the<br>Doctors for Covid Ethics Symposium to the public. <a href="https://t.co/m7HV99vRUe">https://t.co/m7HV99vRUe</a></p>&mdash; Debbee Hutchinson🇬🇧🇬🇧 (@DebbeeHutchins1) <a href="https://twitter.com/DebbeeHutchins1/status/1421221822893277190?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">This is exactly what we need <br>A multi-national study 👇👇<a href="https://twitter.com/hashtag/GCSNeuroCOVID?src=hash&amp;ref_src=twsrc%5Etfw">#GCSNeuroCOVID</a><br>to understand<br>&quot;Neurological manifestations, and the long-term implications of <a href="https://twitter.com/hashtag/COVID19?src=hash&amp;ref_src=twsrc%5Etfw">#COVID19</a> in children and their families&quot; <a href="https://twitter.com/hashtag/pedsICU?src=hash&amp;ref_src=twsrc%5Etfw">#pedsICU</a> <a href="https://twitter.com/hashtag/PICSp?src=hash&amp;ref_src=twsrc%5Etfw">#PICSp</a> <a href="https://twitter.com/hashtag/longcovid?src=hash&amp;ref_src=twsrc%5Etfw">#longcovid</a> <a href="https://twitter.com/PIPSQC?ref_src=twsrc%5Etfw">@PIPSQC</a> <a href="https://twitter.com/DrSevilBG1?ref_src=twsrc%5Etfw">@DrSevilBG1</a> <a href="https://twitter.com/PNCRGtweets?ref_src=twsrc%5Etfw">@PNCRGtweets</a> <a href="https://twitter.com/griz1?ref_src=twsrc%5Etfw">@griz1</a> <a href="https://twitter.com/hashtag/MISC?src=hash&amp;ref_src=twsrc%5Etfw">#MISC</a> <a href="https://t.co/rsCLRqAJTY">https://t.co/rsCLRqAJTY</a> <a href="https://t.co/GAxjUL7off">pic.twitter.com/GAxjUL7off</a></p>&mdash; Yonca Bulut M.D. (@yoncabulutmd) <a href="https://twitter.com/yoncabulutmd/status/1410003565871919105?ref_src=twsrc%5Etfw">June 29, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Yes, he absolutely SHOULD apologise to every single <a href="https://twitter.com/hashtag/LongCovid?src=hash&amp;ref_src=twsrc%5Etfw">#LongCovid</a> patient he has let down.<br><br>But whilst he is at it he must also apologise to the 250k (a VAST under count) <a href="https://twitter.com/hashtag/MECFS?src=hash&amp;ref_src=twsrc%5Etfw">#MECFS</a> patients who have been abandoned and neglected for decades.<a href="https://twitter.com/hashtag/LCandMESolidarity?src=hash&amp;ref_src=twsrc%5Etfw">#LCandMESolidarity</a> <a href="https://t.co/r4BBpja4QE">https://t.co/r4BBpja4QE</a></p>&mdash; Wading through treacle (@kimisgubbed) <a href="https://twitter.com/kimisgubbed/status/1399024870026010631?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Some long Covid sufferers in England waiting months for treatment<br>I guarantee this is happening in Canada <a href="https://t.co/KszG8MoSlg">https://t.co/KszG8MoSlg</a></p>&mdash; 🇨🇦 Merlin 🇨🇦 Wear an N95 #COVIDisAirborne (@MerlinofCanada) <a href="https://twitter.com/MerlinofCanada/status/1399033349306388485?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                </div>
                """,unsafe_allow_html=True)
            
            with col_44:
                # =============== Fear ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Fear😨😱</h4>",unsafe_allow_html=True)
                st.caption("A response to impending danger. It is a survival mechanism that is a reaction to some negative stimulus. It may be a mild caution or an extreme phobia.")
                st.markdown(""" 
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Covid headaches are back... It&#39;s going to be a long night 😩😩</p>&mdash; Todd Ntwana Ya Skontere 🏃🏾 (@Toddinho24) <a href="https://twitter.com/Toddinho24/status/1432423937347100679?ref_src=twsrc%5Etfw">August 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m def scared of vax too cuz Big Pharma has long history of giving me &quot;rare&quot; side effects to point I don&#39;t trust that they are rare or that vax won&#39;t hurt me. Not sure I have enough Covid risk to risk shot but def continually weighing &amp; reevaluating situation</p>&mdash; Lovable_ish🐙#TeamCuttlefish (@lovable_ish) <a href="https://twitter.com/lovable_ish/status/1421206221760180225?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">It sks. My stp son &amp; his wife &amp; son had it. He is now dealing with long Covid &amp; they still don’t know if they want to get vaccinated. 🙄 It gives me a headache worrying abt a few of my fmly members who just won’t get the vaccine. It’s sad but we just stay away from them. 😢😕</p>&mdash; Adnil (@gideja) <a href="https://twitter.com/gideja/status/1421227572969447424?ref_src=twsrc%5Etfw">July 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Why am I so scared right now thinking I’m about to get Covid since I’m at dtf for the first time in a long ass time lol I’m legit freaking out 🥺😫</p>&mdash; janett (@janettsteph7) <a href="https://twitter.com/janettsteph7/status/1408672897112051719?ref_src=twsrc%5Etfw">June 26, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">That was what I feared. I have a long-term patient with CVID plus a natural killer cell defect. I don’t know whether COVID would create severe disease and kill her or just have unfettered replication and form more variants. Obviously, neither outcome would be good. Unfortunately, <a href="https://t.co/E3lPBWzkfH">https://t.co/E3lPBWzkfH</a></p>&mdash; Dr. David Pate (@drpatesblog) <a href="https://twitter.com/drpatesblog/status/1399018551692775429?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sooo... yesterday we found out one of my family members got covid, so everyone got tested today<br>long story short, i&#39;m negative so i dont have covid! yay!<br>i was very scared, yesterday i wasn&#39;t able to sleep well thinking about this, at least im calm rn lol</p>&mdash; F a b u r i n ! 🎀🎄 (@Faburin) <a href="https://twitter.com/Faburin/status/1399144416703623172?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                </div>
                """,unsafe_allow_html=True)
                
                # =============== Anger ===============
                space(1)
                st.markdown("<h4 style='text-align: center; color: black;'>Anger😡</h4>",unsafe_allow_html=True)
                st.caption("Evoked due to injustice, conflict, humiliation, negligence, or betrayal. If anger is active, the individual attacks the target, verbally or physically. If anger is passive, the person silently sulks and feels tension and hostility.")
                st.markdown(""" 
                <div style="width: 100%; height: 400px; overflow-y: scroll">
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">My wife has long covid and has coughed non stop since 7pm, it’s now 3am 👍🏻 fuck off with that shit. <a href="https://twitter.com/hashtag/NewLungs?src=hash&amp;ref_src=twsrc%5Etfw">#NewLungs</a></p>&mdash; Saint🐍 (@POVDenis) <a href="https://twitter.com/POVDenis/status/1432164758397014017?ref_src=twsrc%5Etfw">August 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Get the fucking vax. Wear your mask. <br>I have long covid, 18mths after having covid. Still. <br>Now I have asthma. My lungs are shot. <br>I have neurological issues I didn&#39;t have before. <br>I now have immune issues I didn&#39;t have before.<br>I was healthy. <br>I was active. <br>I was not at risk.</p>&mdash; Snow_Floof_Kat (@Kat_Snow_damnit) <a href="https://twitter.com/Kat_Snow_damnit/status/1432453545299628034?ref_src=twsrc%5Etfw">August 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vax messaging is driving me insane.<br><br>Seeing reports now where folks are talking about a absolutely tiny fraction in a super biased data set of vax people getting long covid.<br><br>SHUT UP. You seriously want to scare people from getting the shot?<br><br>The message is simple. Get vaxxed.</p>&mdash; Will Allred 💜✉️ (@WillAllred117) <a href="https://twitter.com/WillAllred117/status/1420408206006837256?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">He’s blocked me Mark. Fucking moron. I’m 18 months Into long covid and still not recovered. Selfish prick.</p>&mdash; Nick Holder (@valetudocage) <a href="https://twitter.com/valetudocage/status/1420462477591068676?ref_src=twsrc%5Etfw">July 28, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I havent looked at this as it already gets me angry seeing morons like this thinking it&#39;s funny to abuse other people trying to stay safe! Knowing someone I work with recovering with long covid who I would consider fit, this just makes me hope karma will bring these idiots down!</p>&mdash; Dougleelsa26 🏳️‍🌈 (@dougleelsa26) <a href="https://twitter.com/dougleelsa26/status/1399010189181923328?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I’m in very negative spoons today. Literally can’t move at all and have slept all night and all day. Serves me right for walking too far and generally overdoing things yesterday I suppose. I hate long Covid so much. Haven’t managed to do anything I planned to do today. &lt;&lt;sigh&gt;&gt;</p>&mdash; HeliotropeSub (@mauvemaude) <a href="https://twitter.com/mauvemaude/status/1399076205580652544?ref_src=twsrc%5Etfw">May 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

                </div>
                """,unsafe_allow_html=True)


        
    elif choice == "Emotion Predictor":
        add_page_visited_details("Emotion Predictor",datetime.now())
        st.subheader("Emotion Predictor In Text")

        with st.form(key='emotion_form'):
            raw_text = st.text_area('Type Here',"Long Covid brings lots of negative and bad effect to the patient. I feel sorry to those who are suffering from Long Covid symptoms")
            # cleanDocx = cleantext(raw_text)
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            # Apply Prediction Funtion Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
            
            with col2:
                #st.success("Preprocessing Text")
                #st.write(cleanDocx)

                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                porba_df_clean = proba_df.T.reset_index()
                porba_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(porba_df_clean).mark_bar().encode(x='emotions',y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

                st.balloons()  #display some balloons effect xD
                
    
    elif choice == "Monitor":
        add_page_visited_details("Monitor",datetime.now())
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
            st.dataframe(page_visited_details)	

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
            st.altair_chart(c,use_container_width=True)	
            
            p = px.pie(pg_count,values='Counts',names='Pagename')
            st.plotly_chart(p,use_container_width=True)
        
        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
            st.dataframe(df_emotions)
            
            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
            st.altair_chart(pc,use_container_width=True)	
    
    elif choice == "EDA":
        add_page_visited_details("EDA",datetime.now())
        # Read data
        st.title("Exploratory Data Analysis")

        # ----------------- Emotion Metrics Percentage -----------------
        title('Distribution of Emotion',50,'black')
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
        title('Emotions WordCloud',50,'black')

        unique_emotion = ['analytical','neutral','sadness','joy','anger','tentative','fear','confidence']
        st.write(corpus)
        sl = st.slider('Pick Number of Words',50,200)
        
        def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
            return("hsl(240,100%%, %d%%)" % np.random.randint(45,55))
        
        wc = WordCloud(background_color="white", color_func = grey_color_func,max_font_size=150, random_state=42,max_words=sl, collocations=False)

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


        #-------------------------Module 5-----------------------------

        title('Most Popular One Word',50,'black')
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

        #-------------------------Module 6-----------------------------

        title('Most Popular Two Words',50,'black')

        sl_3 = st.slider('Pick Number of Words',5,50,10, key="2")

        # Unigrams - Most Popular One Keyword
        top_text_bigrams = get_top_text_ngrams(corpus.clean_tweet, ngrams=(2,2), nr=sl_3)
        top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
        x, y = zip(*top_text_bigrams)
        bar_C2 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title='Most Popular Two Word', text=y)
        bar_C2.update_traces(textposition="outside", cliponaxis=False)
        bar_C2.update_yaxes(dtick=1, automargin=True)

        st.plotly_chart(bar_C2,use_container_width=True)

        #-------------------------Module 7-----------------------------

        title('Most Popular Three Words',50,'black')

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

        #-------------------------Module 8-----------------------------

        title('Top Keywords For Each Month',50,'black')
        months_name = ['May','June','July','August','September']
        months = {'May':'2021-05','June':'2021-06','July':'2021-07','August':'2021-08','September':'2021-09'}

        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df_date = df.set_index('Datetime')

        col_1,col_2 =st.columns(2)

        with col_1:
            monthChoice = st.radio("Select Month", ('May','June','July','August','September'))
        with col_2:
            sl_5 = st.slider("Pick Number of Words",5,50,10, key="4")

        # Unigrams - Most Popular One Keyword
        selected_month = months[monthChoice]
        top_text_bigrams = get_top_text_ngrams(df_date[selected_month].clean_tweet, ngrams=(1,1), nr=sl_5)
        top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
        x, y = zip(*top_text_bigrams)
        bar_C4 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title=f'Top {sl_5} Keywords In {monthChoice}', text=y)
        bar_C4.update_traces(textposition="outside", cliponaxis=False)
        # bar_C4.update_layout(title=f'Top KeywordWord In{monthChoice}')
        bar_C4.update_yaxes(dtick=1, automargin=True)

        st.plotly_chart(bar_C4,use_container_width=True)

        #----------------------Line Chart Keywords--------------------------
        title('Top 10 Emerging Words',50,'black')
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





    elif choice == "Documentation":
        add_page_visited_details("Documentation",datetime.now())
        st.write('User Guide')

        space(2)

    else:
        st.subheader("About")
        add_page_visited_details("About",datetime.now())





if __name__ == '__main__':
    main()