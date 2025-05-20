#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from glob import glob
import zipfile
from functools import reduce
import os

from utils import value_counts, decorate

# Target variables and their IRNs
TARGET_IRNS = {
    'gender': 30,  # Gender question
    'fework': 7930,  # Men and women should be paid the same money
    'fejob': 7950,   # Women should have same job opportunities
    'fefam': 7940,   # Women should take care of running their homes
    'weight': 5,  # Standard sampling weight
    'grade': 24770    # Grade level
}

# Data file mappings
ZIPFILE = {
    1991: 'data/ICPSR_02521-V2.zip',
    1992: 'data/ICPSR_02522-V2.zip',
    1993: 'data/ICPSR_02523-V1.zip',
    1994: 'data/ICPSR_02475-V1.zip',
    1995: 'data/ICPSR_02390-V1.zip',
    1996: 'data/ICPSR_02350-V2.zip',
    1997: 'data/ICPSR_02476-V1.zip',
    1998: 'data/ICPSR_02752-V2.zip',
    1999: 'data/ICPSR_02940-V1.zip',
    2000: 'data/ICPSR_03183-V1.zip',
    2001: 'data/ICPSR_03426-V1.zip',
    2002: 'data/ICPSR_03752-V2.zip',
    2003: 'data/ICPSR_04018-V2.zip',
    2004: 'data/ICPSR_04263-V2.zip',
    2005: 'data/ICPSR_04537-V2.zip',
    2006: 'data/ICPSR_20180-V2.zip',
    2007: 'data/ICPSR_22500-V1.zip',
    2008: 'data/ICPSR_25422-V2.zip',
    2009: 'data/ICPSR_28402-V1.zip',
    2010: 'data/ICPSR_30984-V1.zip',
    2011: 'data/ICPSR_33902-V1.zip',
    2012: 'data/ICPSR_34574-V2.zip',
    2013: 'data/ICPSR_35166-V2.zip',
    2014: 'data/ICPSR_36149-V1.zip',
    2015: 'data/ICPSR_36407-V1.zip',
    2016: 'data/ICPSR_36799-V1.zip',
    2017: 'data/ICPSR_37183-V1.zip',
    2018: 'data/ICPSR_37415-V1.zip',
    2019: 'data/ICPSR_37842-V1.zip',
    2020: 'data/ICPSR_38189-V1.zip',
    2021: 'data/ICPSR_38502-V1.zip',
    2022: 'data/ICPSR_38883-V1.zip',
    2023: 'data/ICPSR_39171-V1.zip'
}

def read_dta_from_zip(zip_filename, index=0):
    """Read a Stata .dta file from a ZIP archive.
    
    Args:
        zip_filename (str): Path to the ZIP file containing the .dta file
        index (int): Index of the .dta file to read if multiple exist (default: 0)
        
    Returns:
        pandas.DataFrame: The data from the .dta file
        
    Raises:
        FileNotFoundError: If no .dta file is found in the ZIP archive
    """
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # Get the list of files and find the first .dta file
        file_list = zip_ref.namelist()
        dta_files = [f for f in file_list if f.lower().endswith('.dta')]

        if len(dta_files) == 0:
            raise FileNotFoundError("No .dta file found in the ZIP archive.")

        # Read the .dta file into a DataFrame
        stata_path = dta_files[index]
        with zip_ref.open(stata_path) as dta_file:
            df = pd.read_stata(dta_file, convert_categoricals=False)

    return df

def get_irn_mapping(filename='irn_ref_edit.csv'):
    """Get the IRN to variable name mapping.
    
    Args:
        filename (str): Path to the IRN reference file
        
    Returns:
        pandas.DataFrame: DataFrame indexed by ['year', 'irn', 'form'] 
        with columns ['ds', 'V4', 'varname']
    """
    irn_ref = pd.read_csv(filename).set_index(['year', 'irn', 'form'])

    # Add irn 5 rows with mapping info
    years = irn_ref.index.get_level_values('year').unique()
    for year in years:
        ds = irn_ref.loc[(year, 30), 'ds'].iloc[0]
        v4 = irn_ref.loc[(year, 30), 'V4'].iloc[0]
        
        irns = irn_ref.loc[year].index.get_level_values('irn').unique()
        irn = irns[0]  # Just need one to get forms
        forms = irn_ref.loc[(year, irn)].index.get_level_values('form').unique()

        for form in forms:
            irn_ref.loc[(year, 5, form), ['ds', 'V4', 'varname']] = [ds, v4, 'V5']

        irn_ref = irn_ref.sort_index()
    return irn_ref

def read_year_data(year):
    """Read and combine all forms for a given year.
    
    Args:
        year (int): The year to read data for
        
    Returns:
        pandas.DataFrame: Combined data from all forms for the specified year
    """
    # Get the IRN mapping for this year
    irn_to_var = get_irn_mapping()[year]
    
    # Read all forms
    zip_filename = ZIPFILE[year]
    dfs = []
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        dta_files = [f for f in zip_ref.namelist() if f.lower().endswith('.dta')]
        for path in dta_files:
            with zip_ref.open(path) as dta_file:
                df = pd.read_stata(dta_file, convert_categoricals=False)
                dfs.append(df)
    
    # Rename columns in each form to match target variable names
    for irn, vars in irn_to_var.items():
        for df, var in zip(dfs, vars):
            assert var in df.columns
            df.rename(columns={var: irn}, inplace=True)

    columns = ['CASEID'] + list(irn_to_var)
    combined_df = pd.concat(dfs, ignore_index=True)[columns]
    
    return combined_df

def set_target(df, varname, values, newname):
    """Create a binary target variable from a categorical variable.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the source variable
        varname (str): Name of the source variable
        values (list): List of values to consider as True
        newname (str): Name for the new binary variable
    """
    valid = df[varname].notna()
    df[newname] = np.where(valid, df[varname].isin(values), np.nan)

def process_year(year, force_run=False):
    """Process MTF data for a specific year.
    
    This function reads the MTF data for the given year, cleans and processes the variables,
    creates target variables, and saves the processed data to an HDF file.
    
    Args:
        year (int): The year to process data for
        force_run (bool): If True, ignore existing HDF file and reprocess raw data
        
    Returns:
        pandas.DataFrame: The processed dataframe containing cleaned variables and target variables
    """
    hdf_filename = f'mtf_{year}.h5'
    
    # Check if processed data already exists and force_run is False
    if os.path.exists(hdf_filename) and not force_run:
        return pd.read_hdf(hdf_filename, key='data')
    
    df = read_year_data(year)

    # Clean the variables
    clean_df = pd.DataFrame()
    clean_df['grade'] = df['grade'].replace([0, 9, -9], np.nan).replace({2: 8, 4: 10})
    clean_df['gender'] = df['gender'].replace([0, 9, -9], np.nan)
    clean_df['fefam_raw'] = df['fefam'].replace([0, 6, 9, -8, -9], np.nan)
    clean_df['fework_raw'] = df['fework'].replace([0, 9, -8, -9], np.nan)
    clean_df['fejob_raw'] = df['fejob'].replace([0, 9, -8, -9], np.nan)
    clean_df['weight'] = df['weight'].replace([0, 9, -9], np.nan)

    # Save the data in an HDF file
    clean_df.to_hdf(hdf_filename, key='data', mode='w')

    return clean_df

def compute_target_means(df, weighted=False, year=None):
    """Compute means of fejob, fework, and fefam for the given DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the raw variables and weight
        weighted (bool): If True, compute weighted means; else unweighted
        year (int, optional): Year for debugging output
    Returns:
        dict: Dictionary with means for fejob, fework, and fefam
    """
    set_target(df, 'fefam_raw', [1, 2], 'fefam')
    set_target(df, 'fework_raw', [4, 5], 'fework')
    set_target(df, 'fejob_raw', [4, 5], 'fejob')
    
    result = {}
    if weighted:
        for var in ['fework', 'fefam', 'fejob']:
            valid = df.dropna(subset=[var, 'weight'])
            print(f"Year: {year}, Variable: {var}")
            print(f"  Valid rows: {len(valid)}")
            print(f"  Weight sum: {valid['weight'].sum()}")
            print(f"  First few {var} values: {valid[var].head().tolist()}")
            print(f"  First few weight values: {valid['weight'].head().tolist()}")
            result[var] = np.average(valid[var], weights=valid['weight'])
    else:
        result['fework'] = df['fework'].mean()
        result['fefam'] = df['fefam'].mean()
        result['fejob'] = df['fejob'].mean()
    return result

def read_years(years):
    # Loop through all years, compute means, and collect results
    results = []
    for year in years:
        df = process_year(year, force_run=True)
        subset = df.query('gender == 1.0').copy()
        result = compute_target_means(subset, weighted=True, year=year)
        result['year'] = year  # Add year to result after computing means
        results.append(result)

    # Create a DataFrame of the results and print
    results_df = pd.DataFrame(results).set_index('year', drop=True)
    print(results_df)

if __name__ == '__main__':
    irn_ref = get_irn_mapping()
    print(irn_ref)

    #year = 1991
    #df = process_year(year, force_run=True)
    #print(df.head())
    # years = sorted(ZIPFILE.keys())
    # results = read_years(years)
    # print(results)
