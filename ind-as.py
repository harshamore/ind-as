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
        response = openai.Completion.create(
            model="text-davinci-003",  # Using a general model, change if needed
            prompt=prompt,
            max_tokens=50
        )
        # Display the response
        st.write("Response from OpenAI API:")
        st.write(response.choices[0].text.strip())
    except Exception as e:
        # Display error if the API call fails
        st.error(f"Error: {e}")
