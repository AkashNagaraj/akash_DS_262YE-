import heapq
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro
from numpy.random import seed, poisson
import numpy as np


def qqplot():
    sm.qqplot(data1,line="45")
    plt.show()


def shapiro_wilk_test():
    print("p-value of matrix is :",shapiro(data))
    seed(0)
    gfg_data = poisson(5, 200)
    print("p-value on poisson data is :",shapiro(gfg_data))


def box_cox_transformation():
    
    min_element1 = heapq.nsmallest(2,set(data1))[-1]
    for idx,val in enumerate(data1):
        if val==0:
            data1[idx] = np.log(min_element1+1)

    min_element2 = heapq.nsmallest(2,set(data2))[-1]
    for idx,val in enumerate(data2):
        if val==0:
            data2[idx] = np.log(min_element2+1)

    min_element3 = heapq.nsmallest(2,set(data3))[-1]
    for idx,val in enumerate(data3):
        if val==0:
            data3[idx] = np.log(min_element3+1)

    min_element4 = heapq.nsmallest(2,set(data4))[-1]
    for idx,val in enumerate(data4):
        if val==0:
            data4[idx] = np.log(min_element4+1)

    fitted_data1, fitted_lambda1 = stats.boxcox(data1)
    fitted_data2, fitted_lambda2 = stats.boxcox(data2)
    fitted_data3, fitted_lambda3 = stats.boxcox(data3)
    fitted_data4, fitted_lambda4 = stats.boxcox(data4)
   
    assert fitted_lambda1!=0 and fitted_lambda2!=0 and fitted_lambda3!=0 and fitted_lambda4!=0, print("Lambda is zero so for prediction use exp(wt)")

    return fitted_data1, fitted_data2, fitted_data3, fitted_data4


def transform_data():
    
    #print(min(data1), min(data2),min(data3),min(data4))
    fitted_data1, fitted_data2, fitted_data3, fitted_data4 = box_cox_transformation()

    fitted_data1 = fitted_data1.reshape(r,c)
    fitted_data2 = fitted_data2.reshape(r,c)
    fitted_data3 = fitted_data3.reshape(r,c)
    fitted_data4 = fitted_data4.reshape(r,c)
    
    location_dataframe1 = pd.DataFrame(data=np.array(fitted_data1),columns=latitude,index=longitude)
    location_dataframe1.to_csv('../data/_transformed_location_matrix1.csv')
 
    location_dataframe2 = pd.DataFrame(data=np.array(fitted_data2),columns=latitude,index=longitude)
    location_dataframe2.to_csv('../data/_transformed_location_matrix2.csv')
 
    location_dataframe3 = pd.DataFrame(data=np.array(fitted_data3),columns=latitude,index=longitude)
    location_dataframe3.to_csv('../data/_transformed_location_matrix3.csv')

    location_dataframe4 = pd.DataFrame(data=np.array(fitted_data4),columns=latitude,index=longitude)
    location_dataframe4.to_csv('../data/_transformed_location_matrix4.csv')
 
    test_df = pd.read_csv("../data/_transformed_location_matrix1.csv") 
    print(test_df.head
            )

def main():
    
    global r,c
    global latitude, longitude
    global data1, data2, data3, data4

    df1 = pd.read_csv("../data/location_matrix1.csv")
    df2 = pd.read_csv("../data/location_matrix2.csv")
    df3 = pd.read_csv("../data/location_matrix3.csv")
    df4 = pd.read_csv("../data/location_matrix4.csv")
    
    r, c = df1.shape
    c = c - 1 # Remove first column with the latitude data
    latitude = df1.columns.tolist()[1:]
    longitude = df1["Unnamed: 0"].tolist()
    df1 = df1.iloc[:,1:]
    
    #print(df1.head)
    #print("Len of lat :{}, Len of longitude : {}".format(len(latitude),len(longitude)))

    data1 = df1.to_numpy().flatten()
    data2 = df1.to_numpy().flatten()
    data3 = df1.to_numpy().flatten()
    data4 = df1.to_numpy().flatten()

    #qqplot() #The data does not seem to be normally distributed based on QQ plot
    #shapiro_wilk_test() #The p-value is ~0 so the data is not normally distributed  
    transform_data()
   

if __name__=="__main__":
    main()
