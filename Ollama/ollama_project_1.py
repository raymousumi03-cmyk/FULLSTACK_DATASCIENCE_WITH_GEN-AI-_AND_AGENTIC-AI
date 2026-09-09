import streamlit as st
from ollama import Client

# Create Ollama client
client = Client(host="http://localhost:11434")

# Configure page
st.set_page_config(
    page_title="Custom LLM model by Mousumi Ray - Ollama",
    layout="centered"
)

# App title
st.title("Mousumi Ray - Ollama App")

model = st.selectbox(
    "Choose an AI model:",
    [
        "deepseek-r1:1.5b",
        "gemma3:270m"
    ]
)


# User input
prompt = st.text_area("Enter your prompt:", height=200)

# Generate response button
if st.button("Generate Response"):

    # Check empty prompt
    if not prompt.strip():
        st.warning("Please enter a prompt.")

    else:
        # Show loading message
        with st.spinner("Thinking..."):

            # Send prompt to Ollama
            response = client.chat(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

        # Display response
        st.success("Response Generated!")
        st.write(response["message"]["content"])