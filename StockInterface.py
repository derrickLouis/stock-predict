import sys
sys.path.insert(1, 'Ai_Project\TimeSeries')
import TimeSeries as ts
import streamlit as st
import requests

def intro():
    st.title("Welcome to the Stock Prediction Model!")
    st.warning("Ai Stock Prediction models are never 100% true as they can't predict events that may happen in the future.")
intro()

def main():
    st.subheader("Great! Let's Begin.")
    searchInput = st.text_input('Enter your desired stock below.', placeholder="Ex. BA, IBM, WMT, etc..")
    searchList = []
    try:
        #URL Search Builder
        baseurl = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={searchInput}&apikey={ts.api}"
        r = requests.get(baseurl)
        data = r.json()

        try:
            for infoDict in data["bestMatches"]:
                searchList += [infoDict['1. symbol']]
            stock = st.selectbox("Let's make it specific :)", tuple(searchList))
            ts.runModel(stock)

        except:
            st.warning('Sorry No Results!', icon="⚠️")
    except:
        st.warning('Sorry it looks like we\'re experiencing high traffic at the moment! Please try again later.', icon="😢")


agree = st.checkbox("Click to accept this warning, indicating that you understand the model will NOT perfectly predict the future")
if agree:
    main()




