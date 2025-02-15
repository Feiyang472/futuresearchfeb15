import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def generate_favorability_data(topic1, topic2, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    topic1_favorability = np.random.randint(30, 71, size=len(date_range)) / 100
    topic2_favorability = np.random.randint(30, 71, size=len(date_range)) / 100
    
    df = pd.DataFrame({
        'Date': date_range,
        topic1: topic1_favorability,
        topic2: topic2_favorability
    })
    
    return df

st.set_page_config(page_title="Topic Favorability Comparison", layout="wide")

st.title("Topic Favorability Comparison Over Time")

col1, col2 = st.columns(2)

with col1:
    topic1 = st.text_input("Enter the first topic", value="AI")

with col2:
    topic2 = st.text_input("Enter the second topic", value="Blockchain")

start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))

if st.button("Generate Favorability Data"):
    if start_date < end_date:
        df = generate_favorability_data(topic1, topic2, start_date, end_date)
        
        fig = px.line(df, x='Date', y=[topic1, topic2], title=f"Favorability Comparison: {topic1} vs {topic2}")
        fig.update_layout(yaxis_title="Favorability", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Sample Data:")
        st.dataframe(df)
    else:
        st.error("Error: End date must be after the start date.")

st.sidebar.header("About")
st.sidebar.info(
    "This Streamlit app compares the favorability of two topics over time. "
    "Enter the topics you want to compare, select a date range, and click 'Generate Favorability Data' "
    "to see a time series visualization of their relative favorability."
)

