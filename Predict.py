
import pandas as pd
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt

import sys, time

df = pd.read_parquet('./../data/BMTC.parquet.gzip', engine='pyarrow') # This command loads BMTC data into a dataframe. 

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

    
def location_matrix():
    r, c, d = 64, 64, 30 # row and column = 64, distance travelled is 30 meters
    
    for long, lat in ():
        
        # Current longitude, latitude vs next long, lat
        #distance_travelled()

        # Find all BusID in a given longitude and latitude
        all_bus_ID()
        
        # Determine if bus is in motion or not (If 5 consective speed is 0 ignore)
        bus_motion() # get 1/0 values

        # Is the bus in the current location
        start_location_time = ""
        if bus_long<x+1 and bus_long>x and bus_long>y and bus_long<y+1:
            end_location_time = ""

        diff = end_location_time-start_location_time


def find_location(x_start,x_end,y_start,y_end):
    
    temp = pd.DataFrame(df[df['Longitude'].between(x_start,x_end,inclusive="both")]) # Includes the start and end range, add neither to exclude them
    avg = sum(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed'])/len(temp['Speed'])
    
    if avg==0:
        X = temp
        Y = pd.DataFrame(df[df['Latitude'].between(y_start,y_end,inclusive="both")])
        assert sum(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed']) == 0 , "There is data : {} ".format(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed']) 

        #format(set(X['Latitude']).intersection(set(Y['Latitude'])))
    
    del temp

    return round(avg, 4) # Round average minutes to 3 decimals


def location_matrix():
    
    c = 64
    r = 64

    min_latitude, max_latitude = min(df['Latitude']), max(df['Latitude'])
    min_longitude, max_longitude = min(df['Longitude']), max(df['Longitude'])

    lat1, lat2 = min_longitude, max_longitude
    long1, long2 = max_latitude, min_latitude
    
    #print("latitude : {} longitude : {}".format((lat1,lat2),(long1,long2)))
    #top_left, top_right, bottom_left, bottom_right

    horizontal_dist = abs(lat1 - lat2)
    vertical_dist = abs(long1 - long2)

    single_row_size = vertical_dist/r # Divide the heigth by r
    single_col_size = horizontal_dist/c # Divide the width by c

    print("horizontal_dist : {}, vertical_dist : {}, single_row_size : {}, single_col_size : {} ".format(horizontal_dist, vertical_dist, single_row_size, single_col_size))
    
    start = time.time()
    location_matrix = []    
    for i in range(0,c-1): # Traverse 'c' columns
        column_data = []
        for j in range(0,r-1): # Traverse 'r' rows
            val = find_location(lat1+single_col_size*j,lat1+single_col_size*(j+1),long1-single_row_size*(i+1),long1-single_row_size*(i))
            column_data.append(val)
        location_matrix.append(column_data)
    
    print("Time taken to build matrix in seconds : {}".format((time.time()-start)))
    print(location_matrix)

def modelflow():
       
    # Build a matrix 
    location_matrix()

    # Create batches, choose data points only if  

    #print(dfGroundTruth.head(n=5))
    
modelflow()
