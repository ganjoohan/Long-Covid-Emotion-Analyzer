# Core Pkgs
from PIL import Image
img = Image.open("images/webicon.jpg")
import streamlit as st
st.set_page_config(
    page_title="Long Covid Emotion Analyzer",
    page_icon= img,
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### MEsut_GAn"
    }
)

import streamlit.components.v1 as components
from track_utils import create_page_visited_table,create_emotionclf_table
import utils.display as udisp
# import your app modules here
from src import home, dataVisualization, emotionPredictor, monitor, documentation, about

MENU = {
    "Home" : home,
    "Data Visualization" : dataVisualization,
    "Emotion Predictor" : emotionPredictor,
    "Monitor" : monitor,
    "Documentation" : documentation,
    "About" : about,
    
}

def main():
    
    st.sidebar.title("Navigate yourself...")
    menu_selection = st.sidebar.radio("Menu", list(MENU.keys()))

    menu = MENU[menu_selection]

    with st.spinner(f"Loading {menu_selection} ..."):
        udisp.render_page(menu)



if __name__ == '__main__':
    main()