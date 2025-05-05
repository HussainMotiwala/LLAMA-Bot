import streamlit as st
import requests
import time
import json
import os

# Add the serp.ai API functionality
# Add the SerpAPI functionality
def search_web(query, num_results=5):
    """
    Search the web using SerpAPI and return the results
    """
    api_key = os.environ.get("SERPAPI_KEY")  # Get API key from environment variable
    
    if not api_key:
        return {"error": "No SerpAPI key found. Please set the SERPAPI_KEY environment variable."}
    
    # Use the proper SerpAPI endpoint
    url = "https://serpapi.com/search.json"
    
    # Set up parameters according to SerpAPI documentation
    params = {
        "q": query,
        "api_key": api_key,
        "num": num_results,
        "engine": "google"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Format the response to fit our needs
            results = []
            if "organic_results" in data:
                for result in data["organic_results"][:num_results]:
                    results.append({
                        "title": result.get("title", "No title"),
                        "url": result.get("link", "No URL"),
                        "snippet": result.get("snippet", "No snippet")
                    })
            return {"results": results}
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def query_llama_stream(prompt, model="llama3:8b", temperature=0.7, use_web_search=False, search_query=""):
    """
    Query Ollama model with streaming and yield chunks of the response.
    Optionally perform a web search and include results in the prompt.
    """
    # If web search is enabled, perform search and augment prompt
    if use_web_search and search_query:
        search_results = search_web(search_query)
        
        if "error" in search_results:
            error_message = search_results["error"]
            yield f"Web search error: {error_message}", None, False
        else:
            # Format the search results to include in the prompt
            search_content = "Web search results:\n\n"
            for i, result in enumerate(search_results.get("results", [])):
                search_content += f"[{i+1}] {result.get('title', 'No title')}\n"
                search_content += f"URL: {result.get('url', 'No URL')}\n"
                search_content += f"Snippet: {result.get('snippet', 'No snippet')}\n\n"
            
            # Augment the original prompt with search results
            augmented_prompt = f"""
            You have access to recent web search results:
            
            {search_content}
            
            Using this information, please answer the following:
            
            {prompt}
            
            When referencing information from the search results, mention the source.
            """
            prompt = augmented_prompt
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "temperature": temperature
    }
    
    start_time = time.time()
    full_response = ""
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            chunk_text = chunk['response']
                            full_response += chunk_text
                            yield chunk_text, None, False
                        if chunk.get('done', False):
                            duration = time.time() - start_time
                            yield "", duration, True
            else:
                error_text = f"Error {response.status_code}: {response.text}"
                yield error_text, None, True
    except Exception as e:
        error_text = f"Connection Error: {str(e)}"
        yield error_text, None, True

# Set page config
st.set_page_config(page_title="ICICI Prudential Chatbot System", layout="wide")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False

# Main content
st.title("ICICI Prudential Chatbot System")
st.markdown("""
This application uses LLama 3 (8B) through Ollama to answer your questions in a chat-like interface.
It can also search the web for up-to-date information when needed.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    # Add web search toggle
    st.session_state.use_web_search = st.toggle(
        "Enable Web Search",
        value=st.session_state.use_web_search,
        help="When enabled, the chatbot will search the web for relevant information before answering"
    )
    
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.session_state.messages = []
        st.session_state.streaming = False
        st.session_state.current_response = ""
        st.rerun()
    
    st.markdown("---")
    
    
    # Statistics section
    st.header("Statistics")
    if st.session_state.history:
        total_questions = len(st.session_state.history)
        avg_time = sum(item["processing_time"] for item in st.session_state.history) / total_questions
        st.metric("Total Questions", total_questions)
        st.metric("Average Response Time", f"{avg_time:.2f}s")
    
    st.markdown("---")
    st.markdown("© 2025 ICICI Prudential LLM Q&A System")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "time" in message:
                st.caption(f"Response time: {message['time']:.2f}s")

# Handle streaming state
if st.session_state.streaming:
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown(st.session_state.current_response)

# Chat input
if prompt := st.chat_input("Ask your question here", disabled=st.session_state.streaming):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Set streaming to True to prevent multiple inputs
    st.session_state.streaming = True
    st.session_state.current_response = ""
    
    # Indicator for web search if enabled
    if st.session_state.use_web_search:
        with st.chat_message("assistant"):
            st.write("Searching the web for relevant information...")
    
    # Stream the assistant's response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        for chunk, duration, done in query_llama_stream(
            prompt=prompt, 
            model="llama3:8b", 
            temperature=temperature,
            use_web_search=st.session_state.use_web_search,
            search_query=prompt if st.session_state.use_web_search else ""
        ):
            if not done:
                st.session_state.current_response += chunk
                response_placeholder.markdown(st.session_state.current_response)
            else:
                if duration:
                    st.session_state.processing_time = duration
                    st.caption(f"Response time: {duration:.2f}s")
                    
                    # Add to history with web search flag
                    st.session_state.history.append({
                        "question": prompt,
                        "answer": st.session_state.current_response,
                        "temperature": temperature,
                        "web_search_used": st.session_state.use_web_search,
                        "processing_time": duration,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": st.session_state.current_response,
                        "time": duration
                    })
                    
                    # Reset streaming state
                    st.session_state.streaming = False
                    st.rerun()