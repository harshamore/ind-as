import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import os
import pickle

# Define path for AS 21 document and embeddings
as21_pdf_url = "https://resource.cdn.icai.org/69249asb55316-as21.pdf"
embedding_path = "/tmp/as21_embeddings.pkl"

# Step 1: AS21 Compliance Agent
class AS21ComplianceAgent:
    def __init__(self):
        self.chunks, self.embeddings, self.vectorizer = self.load_or_create_embeddings()

    def fetch_and_process_pdf(self, url):
        response = requests.get(url)
        with open("/tmp/as21.pdf", "wb") as f:
            f.write(response.content)

        reader = PdfReader("/tmp/as21.pdf")
        document_text = ""
        for page in reader.pages:
            document_text += page.extract_text()
        return document_text

    def chunk_text(self, text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def create_embeddings(self):
        document_text = self.fetch_and_process_pdf(as21_pdf_url)
        chunks = self.chunk_text(document_text)
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(chunks)
        
        with open(embedding_path, "wb") as f:
            pickle.dump((chunks, embeddings, vectorizer), f)
        return chunks, embeddings, vectorizer

    def load_or_create_embeddings(self):
        if os.path.exists(embedding_path):
            try:
                with open(embedding_path, "rb") as f:
                    chunks, embeddings, vectorizer = pickle.load(f)
                return chunks, embeddings, vectorizer
            except (EOFError, ValueError, pickle.UnpicklingError) as e:
                st.error(f"Failed to load embeddings, recreating: {e}")
                return self.create_embeddings()
        else:
            return self.create_embeddings()

    def retrieve_relevant_info(self, query):
        query_embedding = self.vectorizer.transform([query])
        scores = query_embedding * self.embeddings.T
        best_chunk_idx = scores.toarray().argmax()
        return self.chunks[best_chunk_idx]

# Step 2: Data Entry Agent
class DataEntryAgent:
    def __init__(self, compliance_agent):
        self.compliance_agent = compliance_agent

    def process_excel_files(self, files):
        compliance_query = "What information needs to be checked for compliance with AS 21?"
        required_info = self.compliance_agent.retrieve_relevant_info(compliance_query)

        results = {}
        for file in files:
            df = pd.read_excel(file)
            missing_info = []
            for info in required_info:
                if not df.apply(lambda row: row.astype(str).str.contains(info, regex=False).any(), axis=1).any():
                    missing_info.append(info)
            results[file.name] = missing_info
        return results

# Step 3: Consolidation Agent
class ConsolidationAgent:
    def __init__(self, compliance_agent, data_entry_agent):
        self.compliance_agent = compliance_agent
        self.data_entry_agent = data_entry_agent

    def consolidate(self, files):
        results = self.data_entry_agent.process_excel_files(files)
        consolidated_output = {
            "Compliance Check": results,
            "Summary": "Consolidated statement based on AS 21 compliance checks and data entries."
        }

        # Write results to an Excel file
        with pd.ExcelWriter("/tmp/consolidation_excel.xlsx") as writer:
            for file_name, missing_info in results.items():
                df = pd.DataFrame({"Missing Information": missing_info})
                df.to_excel(writer, sheet_name=file_name[:31], index=False)  # Limit sheet name to 31 chars
            summary_df = pd.DataFrame([{"Summary": consolidated_output["Summary"]}])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        return "/tmp/consolidation_excel.xlsx"

# Instantiate agents
as21_compliance_agent = AS21ComplianceAgent()
data_entry_agent = DataEntryAgent(as21_compliance_agent)
consolidation_agent = ConsolidationAgent(as21_compliance_agent, data_entry_agent)

# Streamlit UI
st.title("AS 21 Compliance & Consolidation Tool")

# Upload multiple Excel files
uploaded_files = st.file_uploader("Upload Financial Statements (Excel)", accept_multiple_files=True, type="xlsx")

# Process files and consolidate data
if st.button("Process & Consolidate"):
    if uploaded_files:
        with st.spinner("Processing and consolidating..."):
            excel_path = consolidation_agent.consolidate(uploaded_files)
            st.success("Consolidation Complete!")
            
            # Download button for the consolidated Excel file
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="Download Consolidated Excel",
                    data=f,
                    file_name="consolidated_accounts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.warning("Please upload at least one file.")
