import streamlit as st
import pandas as pd
import io

def detect_tables(df):
    """
    Detects multiple distinct tables within a single DataFrame based on empty rows and columns.
    Returns a list of DataFrames.
    """
    # 1. Binarize: True if cell has data, False if empty/NaN
    # We treat empty strings or whitespace ONLY strings as empty
    mask = df.notna() & df.apply(lambda x: x.astype(str).str.strip() != "")
    
    # If the sheet is empty, return empty list
    if not mask.any().any():
        return []

    # 2. Find connected components
    rows, cols = df.shape
    visited = set()
    tables = []

    # Directions for adjacency: 8-connectivity to be safe 
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for r in range(rows):
        for c in range(cols):
            if mask.iat[r, c] and (r, c) not in visited:
                # Start a new component
                stack = [(r, c)]
                visited.add((r, c))
                
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    
                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)
                    
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if mask.iat[nr, nc] and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                stack.append((nr, nc))
                
                # Extract the rectangular sub-dataframe
                sub_df = df.iloc[min_r : max_r + 1, min_c : max_c + 1]
                tables.append(sub_df)

    return tables

st.set_page_config(page_title="Excel Table Extractor", layout="wide")

st.title("📊 Excel Table Extractor")
st.markdown("Upload an Excel file containing multiple tables on the same or different sheets. We'll split them for you!")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Read all sheets
        xls = pd.read_excel(uploaded_file, sheet_name=None, header=None)
        
        st.success(f"Successfully loaded {len(xls)} sheets.")
        
        for sheet_name, df in xls.items():
            with st.expander(f"Sheet: {sheet_name}", expanded=True):
                # Detect tables
                detected_tables = detect_tables(df)
                
                if not detected_tables:
                    st.warning("No tables detected in this sheet (or sheet is empty).")
                    continue
                
                st.info(f"Detected {len(detected_tables)} potential separate tables.")
                
                tabs = st.tabs([f"Table {i+1}" for i in range(len(detected_tables))])
                
                for i, (tab, table) in enumerate(zip(tabs, detected_tables)):
                    with tab:
                        # Display
                        st.dataframe(table, use_container_width=True)