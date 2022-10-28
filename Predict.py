
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

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
    # s = int(len(dfOutput)/16)
    print(sum(abs(dfOutput['ETT'] - dfGroundTruth['TT']))/len(dfOutput['ETT']))


def data_summary():
    print(df.head()) #training data
    print(dfInput.head()) #test input
    print(dfGroundTruth.head(n=5)) #test output

def find_location(x_start,x_end,y_start,y_end):
    
    temp = pd.DataFrame(df[df['Longitude'].between(x_start,x_end,inclusive="both")]) # Includes the start and end range, add neither to exclude them
    
    try:
        avg = sum(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed'])/len(temp['Speed'])
    except:
        avg = 0

    if avg==0:
        X = temp
        Y = pd.DataFrame(df[df['Latitude'].between(y_start,y_end,inclusive="both")])
        assert sum(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed']) == 0 , "There is data : {} ".format(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed']) 

    del temp

    return round(avg, 4) # Round average minutes to 3 decimals


def calculate_dimension(r,c):
    min_latitude, max_latitude = min(df['Latitude']), max(df['Latitude'])
    min_longitude, max_longitude = min(df['Longitude']), max(df['Longitude'])

    lat1, lat2 = min_longitude, max_longitude
    long1, long2 = max_latitude, min_latitude

    horizontal_dist = abs(lat1 - lat2)
    vertical_dist = abs(long1 - long2)

    single_row_size = vertical_dist/r # Divide the heigth by r
    single_col_size = horizontal_dist/c # Divide the width by c
    
    return single_row_size, single_col_size


def location_matrix(r=32,c=32):

    min_latitude, max_latitude = min(df['Latitude']), max(df['Latitude'])
    min_longitude, max_longitude = min(df['Longitude']), max(df['Longitude'])

    lat1, lat2 = min_longitude, max_longitude
    long1, long2 = max_latitude, min_latitude
    single_row_size, single_col_size = calculate_dimension(r,c)
    
    start = time.time()
    latitude, location_matrix =  [], []
    for i in range(0,r): # Traverse 'r' rows/ latitudes
        column_data, longitude = [], []
        for j in range(0,c): # Traverse 'c' columns/ longitudes
            val = find_location(lat1+single_col_size*i,lat1+single_col_size*(i+1),long1-single_row_size*(j+1),long1-single_row_size*(j))
            column_data.append(val)
            longitude.append(long1-single_row_size*(j))
        latitude.append(lat1+single_col_size*i)
        location_matrix.append(column_data)

    location_dataframe = pd.DataFrame(data=np.array(location_matrix).transpose(),columns=latitude,index=longitude) #, columns=longitude,index=latitude,inplace=True)
    
    print(location_dataframe.head())
    print("Time taken to build matrix in seconds : {}".format((time.time()-start)))
    location_dataframe.to_csv('../data/location_matrix.csv')
    return single_row_size, single_col_size 


def get_index(x,y):
    
    #print("Searching for location of ",x,y)

    Lat = list(location_df["Unnamed: 0"])
    Long = [float(val) for val in location_df.columns.tolist()[1:]]

    if x<=min(Lat):
        row = 0
    elif x>=max(Lat):
        row = len(Lat)-1 # Give last value
    elif x in Lat:
        row = Lat.index(x)
    else:
        for i in range(len(Lat)-1):
            if Lat[i]>x and Lat[i+1]<x:
                row = i

    if y<=min(Long):
        col=0
    elif y>=max(Long):
        col=len(Long)-1 # Give last value\
    elif y in Long:
        col = Long.index(y) 
    else:
        for j in range(len(Long)-1):
            if Long[j]<y and Long[j+1]>y:
                col = j
    
    try :
        col = col
        row = row
    except:
        print(Lat, Long)
        print(x,y)

    # The position the data lies is between (r,c) , (r+1,c+1)
    
    return row, col


def path_of_points(X,Y):

    location_matrix = np.array(location_df)    
    assert location_matrix.shape == (r,c+1), "The location matrix is of the wrong shape"

    r1,c1 = X[0],X[1]
    r2,c2 = Y[0],Y[1]

    # Adding 1 because first column is latitude values
    c1 = c1+1
    c2 = c2+1

    path = [] # Used to keep track of the datapoints that were used 
    P1,P2=0,0
    try:
        P1 += np.square(location_matrix[r1,c1]*Weight_matrix[r1,c1])
        P2 += np.square(location_matrix[r2,c2]*Weight_matrix[r2,c2])
        path.append((r1,c1))
        path.append((r2,c2))
    except:
        print("Not getting path for this pair : ",(r1,c1),(r2,c2))
        print("Shape of location matrix : {}, weight_matrix : {}".format(location_matrix.shape, Weight_matrix.shape))

    abs_row_val = abs(r1-r2)
    
    for i in range(1,abs_row_val): # Start at 1 because we ignore current postion since it is already in the initialization
        if r1<r2:
            P1 += np.square(location_matrix[r1+i,c1]*Weight_matrix[r1+i,c1])
            P2 += np.square(location_matrix[r2-i,c2]*Weight_matrix[r2-i,c2])
            path.append((r1+i,c1))
            path.append((r2-i,c2))
        elif r1>r2:
            P1 += np.square(location_matrix[r1-i,c1]*Weight_matrix[r1-i,c1])
            P2 += np.square(location_matrix[r2+i,c2]*Weight_matrix[r2+i,c2])
            path.append((r1-i,c1))
            path.append((r2+i,c2))

    abs_col_val = abs(c1-c2)
    for j in range(1,abs_col_val):
        if c1<c2:
            P1 += np.square(location_matrix[r1,c1+j]*Weight_matrix[r1,c1+j])
            P2 += np.square(location_matrix[r2,c2-j]*Weight_matrix[r2,c2-j])
            path.append((r1,c1+j))
            path.append((r2,c2-j))
        elif c1>c2:
            P1 += np.square(location_matrix[r1,c1-j]*Weight_matrix[r1,c1-j])
            P2 += np.square(location_matrix[r2,c2+j]*Weight_matrix[r2,c2+j])
            path.append((r1,c1-j))
            path.append((r2,c2+j))
    
    # Add CNN bias here
    return P1,P2,path
    

def calculate_path(input_data,output):
    
    output = output["TT"]
    loss = 0

    all_paths = []
    for idx,rows in input_data.iterrows():#index
        # The below code calculates the positions in the matrix and then multiplies the weights of them to a weights matrix
        # To improve this we can be selective positions, CNN bias
        S_lat, S_long, D_lat, D_long = rows["Source_Lat"], rows["Source_Long"], rows["Dest_Lat"], rows["Dest_Long"]
        position1 = get_index(S_lat, S_long) 
        position2 = get_index(D_lat, D_long)
        time_P1,time_P2,path = path_of_points(position1,position2)
        time_P = (np.sqrt(time_P1)+np.sqrt(time_P2))/2 
        try:
            loss += abs(time_P - output[idx])
        except:
            print("len input : {}, len output : {} ".format(len(dfInput),len(output)))
        all_paths.append(path)
    return loss, all_paths
    

def update_weight_matrix(positions,lr):
    # Get respective weights from W store them in L
    L = []
    positions = [val for l in positions for val in l]
    for (r,c) in positions:
        L.append(Weight_matrix[r,c])
    # calc gradients and then , new_weights = old_weights - lr*gradients
    gradients = np.gradient(L)
    for idx,(r,c) in enumerate(positions):
        Weight_matrix[r,c] = Weight_matrix[r,c] - lr*gradients[idx]
    # deal with vanishing gradients 


def train_data(r,c):
    
    global Weight_matrix 
    Weight_matrix = np.random.random((r,c+1)) # Since first column in input is not needed

    #features = dfInput[["Source_Lat","Source_Long","Dest_Lat","Dest_Long"]]
    time_taken = dfGroundTruth["TT"]
    
    # Hyperparameters
    batch_size, learning_rate, epochs = 32, 0.05, 200

    global data_size
    train_percent = 80
    data_size = math.ceil(len(dfInput)*train_percent/100)

    for epoch in range(epochs):
        total_losses = []
        for idx in range(0,len(dfInput[:data_size]),batch_size): # Remove slice of 16 afterwards
            loss, used_datapoints = calculate_path(dfInput[idx:idx+batch_size],dfGroundTruth[idx:idx+batch_size])
            current_loss = loss/batch_size
            total_losses.append(current_loss)
            update_weight_matrix(used_datapoints,learning_rate)
        
            #print("Loss rate is : {}".format(current_loss))
        
        if epoch%10==0:
            print("=== Current epoch is : {} ===".format(epoch))
            print("Max loss : {}, min loss : {}".format(max(total_losses),min(total_losses)))
            #print("Total_losses are : {}".format(total_losses))


def test_error_rate():



def modelflow():
    
    #data_summary() 
    
    global r,c 
    r, c = 32, 32

    #location_matrix(c,r) 
    
    global location_df
    location_df = pd.read_csv("../data/location_matrix.csv")

    train_data(r,c)
    test_error_rate()

    #total_loss()


modelflow()
