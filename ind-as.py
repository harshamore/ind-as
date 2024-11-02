import streamlit as st
import openai

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app title
st.title("OpenAI API Connection Test")

# Text input for prompt
prompt = st.text_input("Enter a prompt for OpenAI:", "Say hello to the world!")

# Button to submit prompt and generate response
if st.button("Generate Response"):
    try:
        # Call OpenAI API with the prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using a supported model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        # Display the response
        st.write("Response from OpenAI API:")
        st.write(response.choices[0].message["content"].strip())
    except Exception as e:
        # Display error if the API call fails
        st.error(f"Error: {e}")

# Button to list available models
if st.button("List Available Models"):
    try:
        models = openai.Model.list()
        model_names = [model['id'] for model in models['data']]
        st.write("Available models:")
        st.write(model_names)
    except Exception as e:
        st.error(f"Error: {e}")
