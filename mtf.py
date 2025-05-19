#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from glob import glob
import zipfile
from functools import reduce
import bisect
import os

from utils import value_counts, decorate

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

def read_all_dta_from_zip(zip_filename):
    """Read all Stata .dta files from a ZIP archive.
    
    Args:
        zip_filename (str): Path to the ZIP file containing .dta files
        
    Returns:
        list: List of pandas.DataFrames, one for each .dta file
        
    Raises:
        FileNotFoundError: If no .dta files are found in the ZIP archive
    """
    dfs = []

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        dta_files = [f for f in zip_ref.namelist() if f.lower().endswith('.dta')]

        if not dta_files:
            raise FileNotFoundError("No .dta files found in the ZIP archive.")

        for path in dta_files:
            with zip_ref.open(path) as dta_file:
                df = pd.read_stata(dta_file, convert_categoricals=False)
                dfs.append(df)

    return dfs

def get_year_to_filename(filenames, index=0):
    """Create a mapping from years to ZIP filenames.
    
    Args:
        filenames (list): List of ZIP filenames to process
        index (int): Index of the .dta file to read from each ZIP (default: 0)
        
    Returns:
        dict: Mapping from years to ZIP filenames
        
    Raises:
        ValueError: If a ZIP file contains data from multiple years
    """
    year_to_filename = {}

    for zip_filename in filenames:
        print(zip_filename)
        df = read_dta_from_zip(zip_filename, index=index)

        years = df['V1'].value_counts()
        if len(years) != 1:
            raise ValueError(f"Unexpected number of unique years in V1 for {zip_filename}: {len(years)}")

        year = int(years.index[0])
        if year < 1900:
            year += 1900
        print(year)
        year_to_filename[year] = zip_filename

    return year_to_filename

def read_forms(zip_filename, indices):
    """Read and merge multiple forms from a ZIP file.
    
    Args:
        zip_filename (str): Path to the ZIP file containing the forms
        indices (list): List of indices specifying which .dta files to read
        
    Returns:
        pandas.DataFrame: Merged data from all specified forms
    """
    dfs = [read_dta_from_zip(zip_filename, index=index) for index in indices]
    suffixes = [None, '_y']
    df = reduce(lambda left, right: 
                pd.merge(left, right, on="CASEID", how="outer", suffixes=suffixes), 
                dfs)
    return df

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

def read_year_data(year):
    """Read and combine all forms for a given year.
    
    Args:
        year (int): The year to read data for
        
    Returns:
        pandas.DataFrame: Combined data from all forms for the specified year
        
    Raises:
        KeyError: If the year is not found in ZIPFILE
    """
    zip_filename = ZIPFILE[year]
    dfs = [read_forms(zip_filename, indices) for indices in INDICES[year]]
    return pd.concat(dfs, ignore_index=True).copy()

class YearLookupDict:
    """A dictionary-like class for looking up values by year.
    
    This class provides a way to look up values based on years, with support for
    finding the closest year if an exact match isn't found. It's useful for
    handling variable names that change over time in the MTF dataset.
    
    Attributes:
        _years (tuple): Sorted years
        _values (tuple): Values corresponding to the years
        _default: Default value to return if no year is found
    """
    
    def __init__(self, year_value_pairs, default=None):
        """Initialize the lookup dictionary.
        
        Args:
            year_value_pairs (dict): Mapping from years to values
            default: Default value to return if no year is found (default: None)
        """
        # Sort by year
        self._years, self._values = zip(*sorted(year_value_pairs.items()))
        self._default = default

    def __getitem__(self, year):
        """Look up a value for a given year.
        
        Args:
            year (int): Year to look up
            
        Returns:
            The value corresponding to the closest year
            
        Raises:
            KeyError: If no year is found and no default is set
        """
        i = bisect.bisect_right(self._years, year) - 1
        if i < 0:
            if self._default is not None:
                return self._default
            raise KeyError(f"No entry for year {year} and no default set.")
        return self._values[i]

    def __contains__(self, year):
        """Check if a year exists in the lookup.
        
        Args:
            year (int): Year to check
            
        Returns:
            bool: True if the year exists, False otherwise
        """
        return year in self._years

    def items(self):
        """Get all year-value pairs.
        
        Returns:
            zip: Iterator of (year, value) pairs
        """
        return zip(self._years, self._values)

    def get(self, year, default=None):
        """Get a value for a year, with a default if not found.
        
        Args:
            year (int): Year to look up
            default: Value to return if year is not found (default: None)
            
        Returns:
            The value for the year, or the default if not found
        """
        try:
            return self[year]
        except KeyError:
            return default

# Data file mappings
filenames = glob('data/ICPSR*.zip')

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

# Variable mappings
INDICES = {
    1991: [[0], [2]],
    1992: [[0, 1], [2, 3]],
    1997: [[0, 1], [4, 5]],
    2012: [[0]]
}
INDICES = YearLookupDict(INDICES)

WEIGHT = {
    1991: 'V5',
}
WEIGHT = YearLookupDict(WEIGHT)

GRADE = {
    1991: 'V1101',
    2011: 'V1101',
    2012: 'V501'
}
GRADE = YearLookupDict(GRADE)

GENDER = {
    1991: 'V1226',
    1992: 'V1225',
    1993: 'V1226',
    1994: 'V1227',
    1995: 'V1233',
    1996: 'V1235',
    1998: 'V1233',
    2001: 'V1232',
    2004: 'V1233',
    2006: 'V1246',
    2010: 'V2238',
    2012: 'V7202'
}
GENDER = YearLookupDict(GENDER)

FEFAM = {
    1991: 'V1141',
    1992: 'V2139',
    1995: 'V2140',
    2004: 'V1143',
    2006: 'V2141',
    2009: 'V2142',
    2012: 'V7341'
}
FEFAM = YearLookupDict(FEFAM)

FEWORK = {
    1991: 'V1139',
    1992: 'V2137',
    1995: 'V2138',
    2004: 'V1141',
    2006: 'V2139',
    2009: 'V2140',
    2012: 'V7339',
}
FEWORK = YearLookupDict(FEWORK)

FEJOB = {
    1991: 'V1140',
    1992: 'V2138',
    1995: 'V2139',
    2004: 'V1142',
    2006: 'V2140',
    2009: 'V2141',
    2012: 'V7340'
}
FEJOB = YearLookupDict(FEJOB)

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

    # Create a new dataframe with the variables we want to keep and clean them
    clean_df = pd.DataFrame()
    clean_df['grade'] = df[GRADE[year]].replace([0, 9, -9], np.nan).replace({2: 8, 4: 10})
    clean_df['gender'] = df[GENDER[year]].replace([0, 9, -9], np.nan)
    clean_df['fefam_raw'] = df[FEFAM[year]].replace([0, 6, 9, -8, -9], np.nan)
    clean_df['fework_raw'] = df[FEWORK[year]].replace([0, 9, -8, -9], np.nan)
    clean_df['fejob_raw'] = df[FEJOB[year]].replace([0, 9, -8, -9], np.nan)
    
    # Add weight column
    clean_df['weight'] = df[WEIGHT[year]].replace([0, 9, -9], np.nan)

    # Save the data in an HDF file
    clean_df.to_hdf(hdf_filename, key='data', mode='w')

    return clean_df

def compute_target_means(df, weighted=False):
    """Compute means of fejob, fework, and fefam for the given DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the raw variables and weight
        weighted (bool): If True, compute weighted means; else unweighted
    Returns:
        dict: Dictionary with means for fejob, fework, and fefam
    """
    set_target(df, 'fefam_raw', [1], 'fefam')
    set_target(df, 'fework_raw', [5], 'fework')
    set_target(df, 'fejob_raw', [5], 'fejob')
    
    result = {}
    if weighted:
        for var in ['fework', 'fefam', 'fejob']:
            valid = df.dropna(subset=[var, 'weight'])
            result[var] = np.average(valid[var], weights=valid['weight'])
    else:
        result['fework'] = df['fework'].mean()
        result['fefam'] = df['fefam'].mean()
        result['fejob'] = df['fejob'].mean()
    return result

def main():
    # Loop through all years, compute means, and collect results
    results = []
    for year in sorted(ZIPFILE.keys()):
        df = process_year(year, force_run=True)
        subset = df.query('gender == 1.0').copy()
        result = compute_target_means(subset, weighted=True)
        result['year'] = year  # Add year to result after computing means
        results.append(result)

    # Create a DataFrame of the results and print
    results_df = pd.DataFrame(results).set_index('year', drop=True)
    print(results_df)

if __name__ == '__main__':
    main()
