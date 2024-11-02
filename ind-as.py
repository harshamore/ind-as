import streamlit as st
import pandas as pd
from openpyxl import Workbook
import openai
import os

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app title
st.title("AS 21 Compliance & Consolidation Tool with OpenAI Consultation")

# Upload multiple Excel files
uploaded_files = st.file_uploader("Upload Financial Statements (Excel)", accept_multiple_files=True, type="xlsx")

# Function to consult OpenAI for AS 21 guidance
def consult_openai(query):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        max_tokens=300
    )
    return response.choices[0].message["content"].strip()

# Initialize a variable to track whether consolidation is complete
consolidation_complete = False

# Process and consolidate the data upon button click
if st.button("Process & Consolidate"):
    if uploaded_files:
        # Initialize the workbook for the consolidated data
        consolidated_wb = Workbook()
        consolidated_ws = consolidated_wb.active
        consolidated_ws.title = "Consolidated Financials"
        
        # Initialize a list to store data for consolidation
        summary_data = []
        
        # Define a function to add sheet data with file and sheet information
        def add_sheet_to_summary(sheet_data, file_name, sheet_name):
            for idx, row in sheet_data.iterrows():
                row_data = [file_name, sheet_name] + row.tolist()
                summary_data.append(row_data)
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            xls = pd.ExcelFile(uploaded_file)
            
            # Process each sheet in the file
            for sheet_name in xls.sheet_names:
                sheet_data = xls.parse(sheet_name)
                add_sheet_to_summary(sheet_data, file_name, sheet_name)
        
        # Write consolidated data to the workbook
        header = ["File", "Sheet"] + [f"Column {i}" for i in range(1, len(summary_data[0]) - 1)]
        consolidated_ws.append(header)

        for row in summary_data:
            consolidated_ws.append(row)

        # Save the consolidated workbook
        consolidated_file_path = "/tmp/consolidated_financials_as21.xlsx"
        consolidated_wb.save(consolidated_file_path)
        
        # Provide a download button for the consolidated file
        with open(consolidated_file_path, "rb") as f:
            st.download_button(
                label="Download Consolidated Excel",
                data=f,
                file_name="consolidated_financials_as21.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.success("Consolidation Complete!")
        
        # Set flag to indicate that consolidation is complete
        consolidation_complete = True
    else:
        st.warning("Please upload at least one file.")

# Explanation Button for AS 21 Guidance (enabled after consolidation)
if consolidation_complete:
    if st.button("Explanation"):
        explanation_query = "What are the key steps for consolidating financial statements according to AS 21?"
        explanation_response = consult_openai(explanation_query)
        
        st.subheader("AS 21 Consolidation Steps")
        st.write(explanation_response)
