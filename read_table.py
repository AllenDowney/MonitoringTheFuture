#!/usr/bin/env python
# coding: utf-8

import tabula
import pandas as pd
from pathlib import Path

def process_table(table):
    """Process a single table to handle headers and create two-level columns.
    
    Args:
        table (pd.DataFrame): Raw table from tabula
        
    Returns:
        pd.DataFrame: Processed table with proper headers
    """
    # Get the header row (first row)
    header = table.iloc[0]
    n_cols = len(header)
    n_base = 3  # 'Code', 'Ref. No.', 'Item Name'
    n_years = (n_cols - n_base) // 2  # Number of years (should be 7 for 1991-1997)
    
    # Level 1: 'item' for first 3 columns, 'question' for next n_years, 'variable' for last n_years
    level1 = ['item'] * n_base + ['question'] * n_years + ['variable'] * n_years
    
    # Level 2: column names for item columns, years for question/variable columns
    level2 = ['Code', 'Ref. No.', 'Item Name'] + [str(year) for year in range(1991, 1998)] * 2
    
    # Create multi-level columns
    multi_cols = pd.MultiIndex.from_arrays(
        [level1, level2],
        names=['category', 'field']
    )
    
    # Remove the header row and set the new column names
    table = table.iloc[1:].copy()
    table.columns = multi_cols
    
    return table

def read_table(pdf_path='data/table.pdf'):
    """Read tables from a PDF file using tabula-py.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of pandas DataFrames containing the tables
    """
    # Try different settings
    print("Trying with stream mode...")
    tables_stream = tabula.read_pdf(
        pdf_path,
        pages='all',
        multiple_tables=True,
        lattice=False,
        stream=True,
        guess=True,
        pandas_options={'header': None}
    )
    
    print(f"Found {len(tables_stream)} tables with stream mode")
    
    # Process each table
    processed_tables = []
    for i, table in enumerate(tables_stream):
        print(f"\nProcessing table {i+1}...")
        processed_table = process_table(table)
        processed_tables.append(processed_table)
        
        # Print info about the processed table
        print(f"Shape: {processed_table.shape}")
        print("\nFirst few rows:")
        print(processed_table.head())
        print("\nColumns:")
        print(processed_table.columns)
    
    return processed_tables

if __name__ == '__main__':
    tables = read_table()
    # Concatenate all tables
    concatenated = pd.concat(tables, ignore_index=True)
    # Write to CSV
    concatenated.to_csv('concatenated_tables.csv', index=False)
    print("Concatenated tables written to 'concatenated_tables.csv'.") 