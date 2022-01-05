import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt
import plotly.express as px 
# Track Utils
from track_utils import add_page_visited_details,view_all_page_visited_details,view_all_prediction_details

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

    add_page_visited_details("Monitor",datetime.now())
    st.title("Monitor App")

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