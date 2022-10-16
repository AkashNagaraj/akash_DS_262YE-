"""
This Python code snippet serves two purposes:
1. illustrates how to use relative path
2. provides the template for code submission
ASSUMPTION: 
1. This Python code is present in the folder 'srika_DS_456AB'.
2. BMTC.parquet.gzip, Input.csv, and GroundTruth.csv are present in the folder 'data'
"""
import pandas as pd
# import other packages here
import math
from math import radians, cos, sin, asin, sqrt

df = pd.read_parquet('./../data/BMTC.parquet.gzip', engine='pyarrow') # This command loads BMTC data into a dataframe. 
                                                                      # In case of error, install pyarrow using: 
                                                                      # pip install pyarrow
dfInput = pd.read_csv('./../data/Input.csv')
dfGroundTruth = pd.read_csv('./../data/GroundTruth.csv') 
# NOTE: The file GroundTruth.csv is for participants to assess the performance their own codes

"""
CODE SUBMISSION TEMPLATE
1. The submissions should have the function EstimatedTravelTime().
2. Function arguments:
    a. df: It is a pandas dataframe that contains the data from BMTC.parquet.gzip
    b. dfInput: It is a pandas dataframe that contains the input from Input.csv
3. Returns:
    a. dfOutput: It is a pandas dataframe that contains the output
"""
def Sample_EstimatedTravelTime(df, dfInput): # The output of this function will be evaluated
    # Make changes here. Here is an example model. Participants should come up with their own model.
    avgSpeed = 20 # Assumption that the average speed of a BMTC bus is 20 km/hour
    dfOutput = pd.DataFrame() #
    
    # Estimate the straight line distance in km between the pairs of input coordinates
    dfInput.loc[:,'Distance'] = dfInput[['Source_Lat', 'Source_Long', 'Dest_Lat', 'Dest_Long']].apply(lambda x: distance((x[0],x[1]),(x[2],x[3])),axis=1)
    dfInput.loc[:,'ETT'] = dfInput['Distance']/avgSpeed
    dfOutput = dfInput[['Source_Lat', 'Source_Long', 'Dest_Lat', 'Dest_Long','ETT']]
    return dfOutput 
  
def distance(coord1, coord2):     
    # math module contains a function named radians which converts from degrees to radians.
    lon1 = radians(coord1[1])
    lon2 = radians(coord2[1])
    lat1 = radians(coord1[0])
    lat2 = radians(coord2[0])
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371

    return(c * r)


def total_loss():
    
    dfOutput = Sample_EstimatedTravelTime(df, dfInput)    
    print(sum(abs(dfOutput['ETT'] - dfGroundTruth['TT'])))

    """
    print(df.head(n=5))
    print(dfOutput.head(n=5))
    print(dfGroundTruth.head(n=5))
    """

total_loss()