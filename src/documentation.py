import streamlit as st

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

    st.title("Documentation")

    st.markdown("""
    <h2 style="font-weight:bolder;font-size:30px;color:#216fdb;text-align:left;">The Menu</h2>

    <p>To read more details about this application, can refer to <a target="_blank" href="https://drive.google.com/file/d/1DuWp-kCj483yFLcHv3Gdc_GHLBGDWMyi/view?usp=sharing">user manual.</a></p>

    <h2 style="font-weight:bolder;font-size:25px;color:#216fdb;text-align:left;">Home</h2>

    The `Home` has an emotion analyzer which is a trained machine learning model that is used to predict the emotion of the user input text. You can input your text or tweets into the *textbox* and click the *Analyze* button to generate the analysis result.

    <h2 style="font-weight:bolder;font-size:25px;color:#216fdb;text-align:left;">Exploratory Data Analysis</h2>

    The `Exploratory Data Analysis` explore the dataset by plotting out various insightful visualizations such as bar chart, word cloud, line chart, etc. to better display the data. You can inspect the dataset that are used in this project and some of the sample tweets for each category of emotions. You can find out the distribution of emotions in the dataset, which words are mostly used in each emotion, which word and combination of words are most popular. You can also find out the trendy words based on timeline.

    **IMPORTANT**: It might take some time for the results to load due to the large dataset that is needed to process. 

    <h2 style="font-weight:bolder;font-size:25px;color:#216fdb;text-align:left;">Monitor</h2>

    The `Monitor` collects the inputs text data in emotion analyzer from the user. You can find out the past analyzed text entered by the user in `Home`  and the results.
    
    <h2 style="font-weight:bolder;font-size:25px;color:#216fdb;text-align:left;">About</h2>

    The `About` introduces the concept of *long COVID*, some details about this application, and the information about the developer.

    """,unsafe_allow_html=True)
