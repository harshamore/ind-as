import streamlit as st
import pandas as pd
import openai
from crewai import Crew, Agent
import requests
from bs4 import BeautifulSoup

# Initialize Streamlit app
st.title("Intelligent Ind AS 110 Financial Consolidation Tool")
st.write("Upload the financial statements of multiple subsidiaries, and click 'Process' to get the consolidated statement.")

# Upload multiple Excel files
uploaded_files = st.file_uploader("Upload Financial Statements (Excel)", accept_multiple_files=True, type="xlsx")

# Initialize OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]
crew = Crew()  # Initialize a Crew instance

# Function to fetch the latest Ind AS 110 document
def fetch_latest_ind_as_110():
    url = "https://www.mca.gov.in/bin/ebook/dms/getdocument?doc=ODQzNg%3D%3D&docCategory=Accounting+Standards&type=open"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text
    else:
        st.error("Failed to fetch the latest Ind AS 110 document.")
        return None

# Fetch the latest Ind AS 110 document
ind_as_110_text = fetch_latest_ind_as_110()

# Define agents with enhanced functionalities and GPT-4 guidance
# DataEntryAgent, ReconciliationAgent, ComplianceAgent, and ConsolidationAgent classes defined here...

# Add agents to the crew
crew.add_agent(DataEntryAgent(name="Data Entry"))
crew.add_agent(ReconciliationAgent(name="Reconciliation"))
crew.add_agent(ComplianceAgent(name="Compliance"))
crew.add_agent(ConsolidationAgent(name="Consolidation"))

# Define the process function
def process_files(files):
    data_entries = {}
    for file in files:
        data_entry_agent = crew.get_agent("Data Entry")
        data = data_entry_agent.perform(file)
        if data:
            data_entries.update(data)
    
    reconciliation_agent = crew.get_agent("Reconciliation")
    reconciled_data = reconciliation_agent.perform(data_entries)

    compliance_agent = crew.get_agent("Compliance")
    compliance_report = compliance_agent.perform(data_entries, reconciled_data)

    consolidation_agent = crew.get_agent("Consolidation")
    consolidated_statement = consolidation_agent.perform(compliance_report)

    return consolidated_statement

# Process button
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

