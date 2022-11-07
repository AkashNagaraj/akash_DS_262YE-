from data_transformation import main_transformation_function

import more_itertools
import pandas as pd
import geopy.distance
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import math,csv
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


def data_summary():
    print(df.head()) #training data
    print(dfInput.head()) #test input
    print(dfGroundTruth.head(n=5)) #test output


def find_location(x_start,x_end,y_start,y_end):
    
    print("FInding location of specific points")

    temp = pd.DataFrame(df[df['Longitude'].between(x_start,x_end,inclusive="both")]) # Includes the start and end range, add neither to exclude them    

    # Use avg_time = avg_speed/distance
    avg_speed = sum(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed'])/len(temp['Speed'])
    avg_time_hours = avg_speed / distance
    
    # Four lists to store speeds at different time intervals, hoping to capture variability
    S1,S2,S3,S4=[],[],[],[]
    row = temp[temp['Latitude'].between(y_start,y_end,inclusive="both")][['Timestamp','Speed']]
    for val in row.iterrows():
        time = int(str(val[1]).split()[2].split(":")[0])
        if t1[0]<=time<=t1[1]:
            S1.append(float(str(val[1]).split()[4]))
        elif t2[0]<=time<=t2[1]:
            S2.append(float(str(val[1]).split()[4]))
        elif t3[0]<=time<=t3[1]:
            S3.append(float(str(val[1]).split()[4]))
        else:
            S4.append(float(str(val[1]).split()[4]))
    
   
    avg_t1,avg_t2,avg_t3,avg_t4=0,0,0,0

    if S1:
        avg_t1 = (sum(S1)/len(S1))/distance
    if S2:
        avg_t2 = (sum(S2)/len(S2))/distance
    if S3:
        avg_t3 = (sum(S3)/len(S3))/distance
    if S4:
        avg_t4 = (sum(S4)/len(S4))/distance


    if sum([avg_t1,avg_t2,avg_t3,avg_t4])==0:
        X = temp
        Y = pd.DataFrame(df[df['Latitude'].between(y_start,y_end,inclusive="both")])
        assert sum(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed']) == 0 , "There is data : {} ".format(temp[temp['Latitude'].between(y_start,y_end,inclusive="both")]['Speed']) 

    del temp
    
    #return round(avg_time_hours*60, 4) # Convert hours to minutes ?
    
    return((avg_t1*60,avg_t2*60,avg_t3*60,avg_t4*60))


def convert_coordinates(coords1,coords2):
    return geopy.distance.geodesic(coords1, coords2).km


def get_distance(lat_,long_,height,width):
    d1 = convert_coordinates((lat_,long_),(lat_-height,long_)) # downward distance
    d2 = convert_coordinates((lat_,long_),(lat_,long_+width)) # sideways distance
    return math.sqrt(d1*d1+d2*d2)


def calculate_dimension(r,c):
    min_latitude, max_latitude = min(df['Latitude']), max(df['Latitude'])
    min_longitude, max_longitude = min(df['Longitude']), max(df['Longitude'])

    lat1, lat2 = min_longitude, max_longitude
    long1, long2 = max_latitude, min_latitude

    horizontal_dist = abs(lat1 - lat2)
    vertical_dist = abs(long1 - long2)

    single_row_size = vertical_dist/r # Divide the heigth by r
    single_col_size = horizontal_dist/c # Divide the width by c
   
    global distance
    distance = get_distance(max_latitude,min_longitude,r,c)

    return single_row_size, single_col_size


def build_location_matrix(r=32,c=32):

    print("insde location matrix func")

    min_latitude, max_latitude = min(df['Latitude']), max(df['Latitude'])
    min_longitude, max_longitude = min(df['Longitude']), max(df['Longitude'])

    lat1, lat2 = min_longitude, max_longitude
    long1, long2 = max_latitude, min_latitude
    single_row_size, single_col_size = calculate_dimension(r,c)
    
    start = time.time()
    latitude = []
    location_matrix1, location_matrix2, location_matrix3, location_matrix4 = [],[],[],[]

    global t1,t2,t3
    t1, t2, t3 = (7,10), (12,15), (18,21) # The morning, afternoon and night peak hours
     
    for i in range(0,r): # Traverse 'r' rows/ latitudes
        column_data1, column_data2, column_data3, column_data4, longitude = [], [], [], [], []
        for j in range(0,c): # Traverse 'c' columns/ longitudes
            val1,val2,val3,val4 = find_location(lat1+single_col_size*i,lat1+single_col_size*(i+1),long1-single_row_size*(j+1),long1-single_row_size*(j))
            column_data1.append(val1)
            column_data2.append(val2)
            column_data3.append(val3)
            column_data4.append(val4)

            longitude.append(long1-single_row_size*(j))
        
        location_matrix1.append(column_data1)
        location_matrix2.append(column_data2)
        location_matrix3.append(column_data3)
        location_matrix4.append(column_data4)

        latitude.append(lat1+single_col_size*i)
    
    print("Exited For Loop")
    
    location_dataframe1 = pd.DataFrame(data=np.array(location_matrix1).transpose(),columns=latitude,index=longitude) 
    location_dataframe1.to_csv('../data/location_matrix1.csv')
    del location_dataframe1
    print("Built matrix1")
    
    location_dataframe2 = pd.DataFrame(data=np.array(location_matrix2).transpose(),columns=latitude,index=longitude) 
    location_dataframe2.to_csv('../data/location_matrix2.csv')
    del location_dataframe2
    print("Built matrix2")
    
    location_dataframe3 = pd.DataFrame(data=np.array(location_matrix3).transpose(),columns=latitude,index=longitude)
    location_dataframe3.to_csv('../data/location_matrix3.csv')
    del location_dataframe3
    print("Built matrix3")
    
    location_dataframe4 = pd.DataFrame(data=np.array(location_matrix4).transpose(),columns=latitude,index=longitude)
    location_dataframe4.to_csv('../data/location_matrix4.csv')
    del location_dataframe4
    print("Built matrix4")

    #print(location_dataframe.head())
    print("Time taken to build matrix in seconds : {}".format((time.time()-start)))
    
    return single_row_size, single_col_size 


def get_index(x,y):
    
    #print("Searching for location of ",x,y)

    Lat = list(location_df1["Unnamed: 0"])
    Long = [float(val) for val in location_df1.columns.tolist()[1:]]

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

    # The position the data lies is (r,c)
    return row, col


def time_taken(P1,P2,P3,P4):
    
    time_P1 = (np.sqrt(P1[0])+np.sqrt(P1[1]))/2
    time_P2 = (np.sqrt(P2[0])+np.sqrt(P2[1]))/2
    time_P3 = (np.sqrt(P3[0])+np.sqrt(P3[1]))/2
    time_P4 = (np.sqrt(P4[0])+np.sqrt(P4[1]))/2
    
    #print(time_weights)

    time_P1 *= time_weights[0][0]
    time_P2 *= time_weights[0][1]
    time_P3 *= time_weights[0][2]
    time_P4 *= time_weights[0][3]

    return time_P1+time_P2+time_P3+time_P4


def relu(x):
    return max(0.0,x)


def invboxcox(y,ld):
    if ld == 0:
        return(relu(np.exp(y)))
    else:
        return(relu(np.exp(np.log(ld*y+1)/ld))) 


def box_cox_path_of_points(X,Y):

    location_matrix1 = np.array(location_df1)    
    location_matrix2 = np.array(location_df2)
    location_matrix3 = np.array(location_df3)
    location_matrix4 = np.array(location_df4)

    assert location_matrix1.shape==location_matrix2.shape==location_matrix3.shape==location_matrix4.shape == (r,c+1) , "The location matrix is of the wrong shape"

    r1,c1 = X[0],X[1]
    r2,c2 = Y[0],Y[1]

    # Adding 1 because first column is latitude values
    c1 = c1+1
    c2 = c2+1

    path = [] # Used to keep track of the datapoints that were used 
    
    P11,P12,P21,P22,P31,P32,P41,P42 = 0,0,0,0,0,0,0,0
    

    #l1,l2,l3,l4 = 0.1,0.1,0.1,0.1
    
    try:
        P11 += np.square(invboxcox(location_matrix1[r1,c1]*Weight_matrix1[r1,c1],l1))
        P12 += np.square(invboxcox(location_matrix1[r2,c2]*Weight_matrix1[r2,c2],l1))
        
        P21 += np.square(invboxcox(location_matrix2[r1,c1]*Weight_matrix2[r1,c1],l2))
        P22 += np.square(invboxcox(location_matrix2[r2,c2]*Weight_matrix2[r2,c2],l2))
        
        P31 += np.square(invboxcox(location_matrix3[r1,c1]*Weight_matrix3[r1,c1],l3))
        P32 += np.square(invboxcox(location_matrix3[r2,c2]*Weight_matrix3[r2,c2],l3))

        P41 += np.square(invboxcox(location_matrix4[r1,c1]*Weight_matrix4[r1,c1],l4))
        P42 += np.square(invboxcox(location_matrix4[r2,c2]*Weight_matrix4[r2,c2],l4))

        path.append((r1,c1))
        path.append((r2,c2))
    except:
        print("Not getting path for this pair : ",(r1,c1),(r2,c2))
        print("Shape of location matrix : {}, weight_matrix : {}".format(location_matrix1.shape, Weight_matrix1.shape))

    abs_row_val = abs(r1-r2)
    
    for i in range(1,abs_row_val): # Start at 1 because we ignore current postion since it is already in the initialization
        if r1<r2:
            P11 += np.square(invboxcox(location_matrix1[r1+i,c1]*Weight_matrix1[r1+i,c1],l1))
            P12 += np.square(invboxcox(location_matrix1[r2-i,c2]*Weight_matrix1[r2-i,c2],l1))
            
            P21 += np.square(invboxcox(location_matrix2[r1+i,c1]*Weight_matrix2[r1+i,c1],l2))
            P22 += np.square(invboxcox(location_matrix2[r2-i,c2]*Weight_matrix2[r2-i,c2],l2))
            
            P31 += np.square(invboxcox(location_matrix3[r1+i,c1]*Weight_matrix3[r1+i,c1],l3))
            P32 += np.square(invboxcox(location_matrix3[r2-i,c2]*Weight_matrix3[r2-i,c2],l3))

            P41 += np.square(invboxcox(location_matrix4[r1+i,c1]*Weight_matrix4[r1+i,c1],l4))
            P42 += np.square(invboxcox(location_matrix4[r2-i,c2]*Weight_matrix4[r2-i,c2],l4))

            path.append((r1+i,c1))
            path.append((r2-i,c2))
        elif r1>r2:
            P11 += np.square(invboxcox(location_matrix1[r1-i,c1]*Weight_matrix1[r1-i,c1],l1))
            P12 += np.square(invboxcox(location_matrix1[r2+i,c2]*Weight_matrix1[r2+i,c2],l1))
            
            P21 += np.square(invboxcox(location_matrix2[r1-i,c1]*Weight_matrix2[r1-i,c1],l2))
            P22 += np.square(invboxcox(location_matrix2[r2+i,c2]*Weight_matrix2[r2+i,c2],l2))
            
            P31 += np.square(invboxcox(location_matrix3[r1-i,c1]*Weight_matrix3[r1-i,c1],l3))
            P32 += np.square(invboxcox(location_matrix3[r2+i,c2]*Weight_matrix3[r2+i,c2],l3))
            
            P41 += np.square(invboxcox(location_matrix4[r1-i,c1]*Weight_matrix4[r1-i,c1],l4))
            P42 += np.square(invboxcox(location_matrix4[r2+i,c2]*Weight_matrix4[r2+i,c2],l4))

            path.append((r1-i,c1))
            path.append((r2+i,c2))

    abs_col_val = abs(c1-c2)
    for j in range(1,abs_col_val):
        if c1<c2:
            P11 += np.square(invboxcox(location_matrix1[r1,c1+j]*Weight_matrix1[r1,c1+j],l1))
            P12 += np.square(invboxcox(location_matrix1[r2,c2-j]*Weight_matrix1[r2,c2-j],l1))

            P21 += np.square(invboxcox(location_matrix2[r1,c1+j]*Weight_matrix2[r1,c1+j],l2))
            P22 += np.square(invboxcox(location_matrix2[r2,c2-j]*Weight_matrix2[r2,c2-j],l2))

            P31 += np.square(invboxcox(location_matrix3[r1,c1+j]*Weight_matrix3[r1,c1+j],l3))
            P32 += np.square(invboxcox(location_matrix3[r2,c2-j]*Weight_matrix3[r2,c2-j],l3))

            P41 += np.square(invboxcox(location_matrix4[r1,c1+j]*Weight_matrix4[r1,c1+j],l4))
            P42 += np.square(invboxcox(location_matrix4[r2,c2-j]*Weight_matrix4[r2,c2-j],l4))

            path.append((r1,c1+j))
            path.append((r2,c2-j))
        elif c1>c2:
            P11 += np.square(invboxcox(location_matrix1[r1,c1-j]*Weight_matrix1[r1,c1-j],l1))
            P12 += np.square(invboxcox(location_matrix1[r2,c2+j]*Weight_matrix1[r2,c2+j],l1))
            
            P21 += np.square(invboxcox(location_matrix2[r1,c1-j]*Weight_matrix2[r1,c1-j],l2))
            P22 += np.square(invboxcox(location_matrix2[r2,c2+j]*Weight_matrix2[r2,c2+j],l2))

            P31 += np.square(invboxcox(location_matrix3[r1,c1-j]*Weight_matrix3[r1,c1-j],l3))
            P32 += np.square(invboxcox(location_matrix3[r2,c2+j]*Weight_matrix3[r2,c2+j],l3))

            P41 += np.square(invboxcox(location_matrix4[r1,c1-j]*Weight_matrix4[r1,c1-j],l4))
            P42 += np.square(invboxcox(location_matrix4[r2,c2+j]*Weight_matrix4[r2,c2+j],l4))
            
            path.append((r1,c1-j))
            path.append((r2,c2+j))
    
    # Add CNN bias here
    return(time_taken((P11,P12),(P21,P22),(P31,P32),(P41,P42)), path)
    

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
        combined_time, path = box_cox_path_of_points(position1,position2)
        try:
            loss += abs(combined_time - output[idx])
        except:
            print("len input : {}, len output : {} ".format(len(dfInput),len(output)))
        all_paths.append(path)
    return loss, all_paths
    

def update_weight_matrix(positions,lr):
    # Get respective weights from W store them in L
    L = []
    positions = [val for l in positions for val in l]
    
    # Caclulate specific weights from all weights
    for (r,c) in positions:
        L.append(Weight_matrix1[r,c])
        L.append(Weight_matrix2[r,c])
        L.append(Weight_matrix3[r,c])
        L.append(Weight_matrix4[r,c])
    
    # calc weight gradients and then , new_weights = old_weights - lr*gradients
    gradients = np.gradient(L)
    for idx,(r,c) in enumerate(positions):
        Weight_matrix1[r,c] = Weight_matrix1[r,c] - lr*gradients[idx]
        Weight_matrix2[r,c] = Weight_matrix2[r,c] - lr*gradients[idx]
        Weight_matrix3[r,c] = Weight_matrix3[r,c] - lr*gradients[idx]
        Weight_matrix4[r,c] = Weight_matrix4[r,c] - lr*gradients[idx]
    
    # calc bias gradients and then , new_weights = old_weights - lr*gradients
    gradients_t = np.gradient([val for val in time_weights[0]])
    for idx,val in enumerate(time_weights[0]):
        time_weights[0][idx] = time_weights[0][idx] - lr*gradients_t[idx]

    # deal with vanishing gradients 

def get_error_prone_paths(all_error_prone_paths): 
    
    print("Inside path function")

    set_data_points = {}
    for l in all_error_prone_paths:
        for points in l:
            for point in points:
                if point not in set_data_points:
                    set_data_points[point]=1
                else:
                    set_data_points[point]+=1
    
    X,Y = [], []
    for key,data in set_data_points.items():
        if data>30:
            X.append(key[0])
            Y.append(key[1])
    
    plt.scatter(X, Y, c ="blue")
    plt.show()

def train_data(r,c):
     
    global Weight_matrix1, Weight_matrix2, Weight_matrix3, Weight_matrix4
    Weight_matrix1 = np.random.random((r,c+1)) # Since first column in input is not needed
    Weight_matrix2 = np.random.random((r,c+1))
    Weight_matrix3 = np.random.random((r,c+1))
    Weight_matrix4 = np.random.random((r,c+1))
    
    global time_weights # Since afternoon, evening and night could have different impact on the vehicles speed and hence time taken
    time_weights = np.random.random((1,4))

    #features = dfInput[["Source_Lat","Source_Long","Dest_Lat","Dest_Long"]]
    time_taken = dfGroundTruth["TT"]
    
    # Hyperparameters
    global batch_size
    batch_size, learning_rate, epochs = 64, 0.001, 100 # Choose parameters with minimum loss

    global data_size
    train_percent = 80
    data_size = math.ceil(len(dfInput)*train_percent/100)

    all_error_prone_paths = []
    
    for epoch in range(epochs):    
        overall_max_loss = 0
        total_losses, error_prone_path = [], []
        for idx in range(0,len(dfInput[:data_size]),batch_size): 
            loss, used_datapoints = calculate_path(dfInput[idx:idx+batch_size],dfGroundTruth[idx:idx+batch_size])
            current_loss = loss/batch_size
            total_losses.append(current_loss)
            update_weight_matrix(used_datapoints,learning_rate)
            #print("Loss rate is : {}".format(current_loss))
        
        if epoch%5==0:
            print("=== Current epoch is : {} ===".format(epoch))
            print("Max loss : {}, min loss : {}".format(max(total_losses),min(total_losses)))
            #print("Total_losses are : {}".format(total_losses))
        
        if overall_max_loss<current_loss:
            error_prone_path = used_datapoints
        
        all_error_prone_paths.append(error_prone_path)
    
    #get_error_prone_paths(all_error_prone_paths)


def estimated_time_travelled(df, dfInput):
    
    total_loss = 0
    for idx in range(data_size,len(dfInput),batch_size):
        loss, used_datapoints = calculate_path(dfInput[idx:idx+batch_size],dfGroundTruth[idx:idx+batch_size])
        total_loss += loss 
    print("The total L1 loss is : ",total_loss/len(dfInput[data_size:]))


def store_matrix_check():
    np.savetxt('stored_weights/Final_W1.csv',Weight_matrix1,delimiter=",")
    np.savetxt('stored_weights/Final_W2.csv',Weight_matrix2,delimiter=",")
    np.savetxt('stored_weights/Final_W3.csv',Weight_matrix3,delimiter=",")
    np.savetxt('stored_weights/Final_W4.csv',Weight_matrix4,delimiter=",")


def modelflow():
    
    #data_summary() 
    
    global r,c 
    r, c = 32,32
    
    #build_location_matrix(c,r) # avg time to build is 1.4 hours
    
    # Transform to normal distribution with box_cox
    global l1,l2,l3,l4
    l1,l2,l3,l4 = main_transformation_function()
    l_df = pd.DataFrame([l1,l2,l3,l4])
    l_df.to_csv("stored_weights/lambda_values.csv")

    global location_df1, location_df2, location_df3, location_df4
    location_df1 = pd.read_csv("../data/location_matrix1.csv")
    location_df2 = pd.read_csv("../data/location_matrix2.csv")
    location_df3 = pd.read_csv("../data/location_matrix3.csv")
    location_df4 = pd.read_csv("../data/location_matrix4.csv")

    """    
    location_df1 = pd.read_csv("../data/_transformed_location_matrix1.csv")
    location_df2 = pd.read_csv("../data/_transformed_location_matrix2.csv")
    location_df3 = pd.read_csv("../data/_transformed_location_matrix3.csv")
    location_df4 = pd.read_csv("../data/_transformed_location_matrix4.csv")
    """
    
    train_data(r,c)
    estimated_time_travelled(df,dfInput) 
    
    store_matrix_check()
    

modelflow()
