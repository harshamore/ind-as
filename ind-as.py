import streamlit as st
import pandas as pd
import openai
from crewai import Crew, Agent
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app title
st.title("Intelligent Ind AS Financial Consolidation Tool with Guide Agent")

# Dropdown for standard selection
selected_standard = st.selectbox("Select Accounting Standard", ["Ind AS 110", "Ind AS 21"])

# Upload multiple Excel files
uploaded_files = st.file_uploader("Upload Financial Statements (Excel)", accept_multiple_files=True, type="xlsx")

# Define GuideAgent to process Ind AS 110 document and create a list of steps
class GuideAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = self.load_or_process_steps()

    def load_or_process_steps(self):
        # Check if steps are cached
        if os.path.exists("/tmp/ind_as_110_steps.txt"):
            with open("/tmp/ind_as_110_steps.txt", "r") as file:
                steps = file.readlines()
            return [step.strip() for step in steps]
        else:
            steps = self.process_ind_as_110_document()
            with open("/tmp/ind_as_110_steps.txt", "w") as file:
                file.write("\n".join(steps))
            return steps

    def fetch_document(self, url):
        response = requests.get(url)
        with open("/tmp/ind_as_110.pdf", "wb") as f:
            f.write(response.content)

    def extract_text_from_pdf(self, filepath):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def split_text_into_chunks(self, text, max_tokens=3000):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_length += len(word) + 1  # Adding 1 for the space
            current_chunk.append(word)
            if current_length >= max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def process_ind_as_110_document(self):
        url = "https://resource.cdn.icai.org/53621asb43065.pdf"
        self.fetch_document(url)
        document_text = self.extract_text_from_pdf("/tmp/ind_as_110.pdf")

        # Split document text into manageable chunks
        chunks = self.split_text_into_chunks(document_text, max_tokens=3000)

        steps = []
        for chunk in chunks:
            prompt = (
                "Extract and summarize the key steps for preparing consolidated financial statements as per Ind AS 110 "
                "from the following text:\n\n" + chunk
            )
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                steps_chunk = response.choices[0].message["content"].strip().splitlines()
                steps.extend(steps_chunk)
            except openai.error.InvalidRequestError as e:
                st.error(f"Invalid request error: {e}")
                return ["Error processing document. Please check your OpenAI API configuration."]
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return ["Error processing document. Please try again later."]

        return steps

    def get_steps(self):
        return self.steps

# Initialize the GuideAgent to create instructions based on Ind AS 110
guide_agent = GuideAgent(name="Guide", role="Guide Specialist", goal="Provide steps for Ind AS 110", backstory="Expert in understanding Ind AS 110.")

# Define other agents with enhanced functionalities, consulting GuideAgent steps as needed
class DataEntryAgent(Agent):
    def perform(self, data, guide_agent):
        steps = guide_agent.get_steps()  # Consult GuideAgent for instructions
        st.write("Steps from GuideAgent for Data Entry:")
        st.write(steps)

        # GPT-4 to interpret selected Ind AS structure and required items
        required_items = self.get_required_items_from_gpt4(selected_standard)
        try:
            extracted_data = pd.read_excel(data, sheet_name=None)
            extracted_trial_balances = {sheet: df for sheet, df in extracted_data.items() if 'Trial Balance' in sheet}

            for sheet, df in extracted_trial_balances.items():
                missing_entries = self.check_missing_entries(df, required_items)
                if missing_entries:
                    handling_guidance = self.get_handling_guidance_from_gpt4(missing_entries, selected_standard)
                    df['Missing Entries'] = ', '.join(missing_entries)
                    df['Handling Guidance'] = handling_guidance
                extracted_trial_balances[sheet] = df

            return extracted_trial_balances
        except Exception as e:
            st.error(f"Data extraction failed: {e}")
            return None

    def get_required_items_from_gpt4(self, standard):
        prompt = f"Analyze the latest {standard} standard and list all required items and their structure for financial statements."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message["content"].strip().split(',')

    def check_missing_entries(self, data, required_items):
        missing_entries = [item for item in required_items if not data['Account'].str.contains(item).any()]
        return missing_entries

    def get_handling_guidance_from_gpt4(self, missing_entries, standard):
        prompt = f"Based on {standard}, how should we proceed if the following items are missing in financial statements: {', '.join(missing_entries)}?"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message["content"].strip()

# Initialize Crew with the defined agents, including GuideAgent
crew = Crew(agents=[
    guide_agent,
    DataEntryAgent(name="Data Entry", role="Data Entry Specialist", goal="Ensure data completeness based on Ind AS standards", backstory="Expert in data validation for financial records."),
])

# Define the process function
def process_files(files):
    data_entries = {}
    for file in files:
        data_entry_agent = crew.get_agent("Data Entry")
        data = data_entry_agent.perform(file, guide_agent)  # Pass GuideAgent to consult instructions
        if data:
            data_entries.update(data)
    
    # Continue with other agents following GuideAgent's steps...
    # (Implementation for ReconciliationAgent, ComplianceAgent, ConsolidationAgent, consulting GuideAgent as needed)

    return data_entries  # Assuming final data

# Process button for actual data processing
if st.button("Process"):
    if uploaded_files:
        with st.spinner("Processing..."):
            final_consolidated_statement = process_files(uploaded_files)
            st.success("Consolidation Complete!")
            st.write("Final Consolidated Statement:")
            st.dataframe(final_consolidated_statement)
            csv = final_consolidated_statement.to_csv(index=False)
            st.download_button("Download Consolidated Statement as CSV", data=csv, file_name="consolidated_statement.csv")
    else:
        st.warning("Please upload at least one file.")

# Test OpenAI Connection button
if st.button("Test OpenAI Connection"):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, OpenAI!"}],
            max_tokens=10
        )
        st.write("Response from OpenAI API:")
        st.write(response.choices[0].message["content"].strip())
    except Exception as e:
        st.error(f"Error: {e}")
