import streamlit as st
import pandas as pd
import openai
from crewai import Crew, Agent
import requests
from bs4 import BeautifulSoup

# Initialize Streamlit app
st.title("Intelligent Ind AS Financial Consolidation Tool")
st.write("Upload the financial statements of multiple subsidiaries, and select the applicable standard.")

# Dropdown for standard selection
selected_standard = st.selectbox("Select Accounting Standard", ["Ind AS 110", "Ind AS 21"])

# Upload multiple Excel files
uploaded_files = st.file_uploader("Upload Financial Statements (Excel)", accept_multiple_files=True, type="xlsx")

# Use OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Access the API key securely

# Function to fetch the latest Ind AS document based on selection
def fetch_latest_ind_as_document(standard):
    if standard == "Ind AS 110":
        url = "https://www.mca.gov.in/bin/ebook/dms/getdocument?doc=ODQzNg%3D%3D&docCategory=Accounting+Standards&type=open"
    elif standard == "Ind AS 21":
        url = "https://www.mca.gov.in/bin/ebook/dms/getdocument?doc=ODQyNg%3D%3D&docCategory=Accounting+Standards&type=open"
    
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text
    else:
        st.error(f"Failed to fetch the latest {standard} document.")
        return None

# Fetch the latest Ind AS document based on user selection
ind_as_text = fetch_latest_ind_as_document(selected_standard)

# Define agents with enhanced functionalities and GPT-4 guidance for both Ind AS 110 and Ind AS 21

class DataEntryAgent(Agent):
    def perform(self, data):
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
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip().split(',')

    def check_missing_entries(self, data, required_items):
        missing_entries = [item for item in required_items if not data['Account'].str.contains(item).any()]
        return missing_entries

    def get_handling_guidance_from_gpt4(self, missing_entries, standard):
        prompt = f"Based on {standard}, how should we proceed if the following items are missing in financial statements: {', '.join(missing_entries)}?"
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

class ReconciliationAgent(Agent):
    def perform(self, data):
        reconciliation_criteria = self.get_reconciliation_criteria_from_gpt4(selected_standard)
        reconciled_data = {sheet: self.reconcile_intercompany(df, reconciliation_criteria) for sheet, df in data.items()}
        return reconciled_data

    def get_reconciliation_criteria_from_gpt4(self, standard):
        prompt = f"Using {standard}, what criteria should be used to reconcile intercompany transactions and accounts?"
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip().split(',')

    def reconcile_intercompany(self, data, reconciliation_criteria):
        intercompany_accounts = data[data['Account Type'] == 'Intercompany']
        elimination_entries = []

        for _, row in intercompany_accounts.iterrows():
            match = data[
                (data['Counterparty Account'] == row['Account']) &
                (data['Amount'] == -row['Amount'])
            ]
            if not match.empty:
                data.loc[data['Account'] == row['Account'], 'Elimination'] = True
                data.loc[data['Counterparty Account'] == row['Counterparty Account'], 'Elimination'] = True
            else:
                explanation = self.generate_discrepancy_explanation(row, reconciliation_criteria, selected_standard)
                elimination_entries.append((row['Account'], explanation))

        data['Unmatched Entries'] = ', '.join([entry[0] for entry in elimination_entries])
        data['Discrepancy Explanation'] = '\n'.join([entry[1] for entry in elimination_entries])
        return data

    def generate_discrepancy_explanation(self, row, criteria, standard):
        prompt = f"Explain why account {row['Account']} with amount {row['Amount']} might not reconcile based on these criteria from {standard}: {', '.join(criteria)}"
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

class ComplianceAgent(Agent):
    def perform(self, data_entries, reconciled_data):
        compliance_report = {}
        for sheet, data in data_entries.items():
            compliance_report[sheet] = {
                'data_entry_compliance': self.verify_data_entry_compliance(data, selected_standard),
                'reconciliation_compliance': self.verify_reconciliation_compliance(reconciled_data[sheet], selected_standard)
            }
        return compliance_report

    def verify_data_entry_compliance(self, data, standard):
        prompt = f"Using {standard}, verify if the data entry process aligns with the required standards and identify any discrepancies."
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

    def verify_reconciliation_compliance(self, data, standard):
        prompt = f"Using {standard}, verify if the intercompany reconciliation aligns with the required standards and identify any discrepancies."
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

class ConsolidationAgent(Agent):
    def perform(self, data):
        consolidated_data = self.consolidate(data)
        consolidated_data = self.eliminate_intercompany_transactions(consolidated_data)
        consolidated_data = self.calculate_nci_and_goodwill(consolidated_data, selected_standard)
        consolidated_data['Summary'] = self.generate_gpt4_summary(consolidated_data, selected_standard)
        return consolidated_data

    def consolidate(self, data):
        consolidated = pd.concat(data.values())
        consolidated['Amount'] = consolidated.groupby('Account')['Amount'].transform('sum')
        consolidated.drop_duplicates(subset='Account', keep='first', inplace=True)
        return consolidated

    def eliminate_intercompany_transactions(self, data):
        data = data[~data['Elimination'].fillna(False)]
        return data

    def calculate_nci_and_goodwill(self, data, standard):
        if 'Non-controlling Interests' in data.columns:
            nci_percentage = data['NCI Percentage'].iloc[0] if 'NCI Percentage' in data.columns else 0.0
            data['NCI Amount'] = data['Amount'] * (nci_percentage / 100)

        if 'Goodwill' not in data.columns:
            data['Goodwill'] = self.calculate_goodwill(data, standard)
        
        return data

    def calculate_goodwill(self, data, standard):
        prompt = f"Using {standard}, calculate goodwill based on the purchase price and fair value of net assets."
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return float(response.choices[0].text.strip())

    def generate_gpt4_summary(self, consolidated_data, standard):
        prompt = f"Provide a summary of the consolidated financial statement under {standard}, covering key aspects like intercompany eliminations, NCI, and goodwill calculations."
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

# Initialize Crew with the defined agents
crew = Crew(agents=[
    DataEntryAgent(name="Data Entry"),
    ReconciliationAgent(name="Reconciliation"),
    ComplianceAgent(name="Compliance"),
    ConsolidationAgent(name="Consolidation")
])

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
