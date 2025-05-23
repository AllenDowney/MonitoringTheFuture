#!/usr/bin/env python
# coding: utf-8


from pathlib import Path

import pandas as pd
import tabula


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
    tables_stream = tabula.read_pdf(
        pdf_path,
        pages='all',
        multiple_tables=True,
        lattice=False,
        stream=True,
        guess=True,
        pandas_options={'header': None}
    )
    
    # Process each table
    processed_tables = []
    for i, table in enumerate(tables_stream):
        processed_table = process_table(table)
        processed_tables.append(processed_table)
    
    # Concatenate all tables
    concatenated = pd.concat(processed_tables, ignore_index=True)
    return concatenated
    

def get_dataset_mappings():
    """Read dataset mappings from CSV file.
    
    Returns:
        tuple of dictionaries: year_to_dataset, dataset_to_year
    """
    df = pd.read_csv('year_dataset_mapping.csv')
    year_to_dataset = dict(zip(df['year'], df['dataset_id']))
    dataset_to_year = dict(zip(df['dataset_id'], df['year']))
    return year_to_dataset, dataset_to_year


def process_icpsr_directories():
    """Process ICPSR directories to infer year and form from the directory structure.
    
    Returns:
        pd.DataFrame: DataFrame containing directory, year, form, subdir, and dta file information
    """
    # Get dataset mappings
    year_to_dataset, dataset_to_year = get_dataset_mappings()
    
    # Base directory for ICPSR data
    base_dir = Path('data')
    
    # List to store results
    results = []

    # Loop through each ICPSR directory
    for dir_name in base_dir.glob('ICPSR_*'):
        if dir_name.is_dir():
            # Extract year using reverse lookup
            year = dataset_to_year.get(dir_name.name, 'Unknown')

            # Loop through subdirectories
            for subdir in dir_name.glob('DS*'):
                if subdir.is_dir():
                    
                    # Look for .dta files in the subdirectory
                    for dta_file in subdir.glob('**/*.dta'):
                        # Get relative path from base_dir
                        rel_path = dta_file.relative_to(base_dir)
                        results.append({
                            'year': year,
                            'directory': dir_name.name,
                            'subdir': subdir.name,
                            'path': str(rel_path)
                        })
    
    # Create DataFrame from results
    df = pd.DataFrame(results).sort_values(by=['year', 'subdir'])
    df = add_form_and_grade(df)
    return df

def add_form_and_grade(df):
    """Add form and grade level columns based on subdirectory counts per year.
    
    Args:
        df (pd.DataFrame): DataFrame from process_icpsr_directories
        
    Returns:
        pd.DataFrame: DataFrame with added form and grade columns
    """
    # Initialize form and grade columns
    df['form'] = None
    df['grade'] = None
    
    # Process each year
    for year, group in df.groupby('year'):
        indices = group.index

        if len(group) == 1:
            df.loc[indices, 'form'] = [1]
            df.loc[indices, 'grade'] = ['8th+10th']
        elif len(group) == 2:
            df.loc[indices, 'form'] = [1,2]
            df.loc[indices, 'grade'] = ['8th','10th']
        elif len(group) == 4:
            df.loc[indices, 'form'] = [1,2,1,2]
            df.loc[indices, 'grade'] = ['8th','8th','10th','10th']
        elif len(group) == 8:
            df.loc[indices, 'form'] = [1,2,3,4,1,2,3,4]
            df.loc[indices, 'grade'] = ['8th','8th','8th','8th','10th','10th','10th','10th']
        else:
            print(year, len(group))
            
    return df

def create_irn_ref_table(df):
    """Create a DataFrame similar to irn_ref_edit.csv from the existing DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame created by read_table

    Returns:
        pd.DataFrame: A new DataFrame with columns ds, year, form, V4, varname, irn, item_name
    """
    year_to_dataset, _ = get_dataset_mappings()

    # Initialize a list to store tuples for the new DataFrame
    data_tuples = []
    
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Extract the year from the column names
        for year in range(1991, 2024):
            if str(year) in df.columns.get_level_values('field'):
                variable = row[('variable', str(year))]
                # Skip if variable is NaN
                if pd.isna(variable):
                    continue
                ds = year_to_dataset[year]
                # Extract form number from the first digit of the variable value
                var_value = str(int(variable))
                varname = f'V{var_value}'
                irn = row[('item', 'Ref. No.')]
                item_name = row[('item', 'Item Name')]
                
                # Append a tuple of the new row data
                data_tuples.append((year, ds, irn, varname, item_name))
    
    # Create the new DataFrame
    new_df = pd.DataFrame(data_tuples, columns=['year', 'ds', 'irn', 'varname', 'item_name'])
    
    return new_df


def save_to_hdf():
    """Read tables from PDF and save the concatenated table to HDF."""
    table = read_table()
    table.to_hdf('cross_time_index.h5', key='table', mode='w')
    print("Saved concatenated table to cross_time_index.h5")

def generate_irn_ref():
    """Read the concatenated table from HDF and generate the irn_ref-style CSV."""
    table = pd.read_hdf('cross_time_index.h5', key='table')
    print("Read table back from cross_time_index.h5")
    new_table = create_irn_ref_table(table)
    print(new_table)
    new_table.to_csv('irn_ref_generated.csv', index=False)
    print("Saved to irn_ref_generated.csv")

if __name__ == '__main__':    
    # Process directories and display results
    dir_df = process_icpsr_directories()
    dir_df.to_csv('icpsr_directories.csv', index=False)
    
    generate_irn_ref()

