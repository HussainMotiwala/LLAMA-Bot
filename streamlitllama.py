# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:38:08 2025

@author: hussa
"""

import streamlit as st
import requests
import time

def query_llama(prompt, model="llama3:8b", temperature=0.7):
    """
    Query Ollama model and return the result.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    
    start_time = time.time()
    
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
        else:
            response_text = f"Error {response.status_code}: {response.text}"
    except Exception as e:
        response_text = f"Connection Error: {str(e)}"
        
    duration = time.time() - start_time
    
        
    return response_text, duration

# Set page config
st.set_page_config(page_title="LLM Q&A Application", layout="wide")

# Main content
st.title("ICICI Prudential Question-Answering System")
st.markdown("""
This application uses LLama 3 (8B) through Ollama to answer your questions.
""")

# Initialize history in session state if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Temperature slider
temperature = st.slider(
    "Temperature", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.7, 
    step=0.1,
    help="Higher values make output more random, lower values more deterministic"
)

# Input area
prompt = st.text_area("Enter your question:", height=100)
col1, col2 = st.columns([1, 5])
submit = col1.button("Submit Question")
clear = col2.button("Clear History")

# Handle clear history
if clear:
    st.session_state.history = []
    st.experimental_rerun()

# Process input and display answer
if submit and prompt:
    with st.spinner('Processing your question...'):
        response, duration = query_llama(prompt, "llama3:8b", temperature)
        
        # Add to history
        st.session_state.history.append({
            "question": prompt,
            "answer": response,
            "temperature": temperature,
            "processing_time": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Show final processing time
        st.success(f"✅ Query completed in {duration:.2f} seconds")

# Display history
if st.session_state.history:
    st.markdown("## Question History")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Q: {item['question'][:50]}... (Time: {item['timestamp']} | Duration: {item['processing_time']:.2f}s | Temp: {item['temperature']})"):
            st.markdown("Question")
            st.write(item["question"])
            
            st.markdown("Answer")
            st.write(item["answer"])

# Footer
st.markdown("---")
st.markdown("© 2025 ICICI Prudential LLM Q&A System")
