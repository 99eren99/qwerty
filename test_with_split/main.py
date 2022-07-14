from cmath import inf
from logging import exception
import time
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

def mse(predicted,real):
    return mean_squared_error(predicted,real)

def mape(pred,actual): 
    return mean_absolute_error(pred,actual)

def first_method(x,y):
    
    start=time.time()
    X=x
    Y = np.log(y)
    inf_index=np.where(np.isinf(Y))
    if 0<len(inf_index[0]):
        print("Dropped infinites for lineer:",len(inf_index[0]))
    X=np.delete(X,inf_index).reshape(-1,1)
    Y=np.delete(Y,inf_index).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(
                             X, Y, test_size=0.33, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    predictions=reg.predict(X_test)
    predictions=np.squeeze(predictions.reshape(1,-1))
    y_test=np.squeeze(np.exp(y_test.reshape(1,-1)))
    end=time.time()
    mse1=mse(predictions,y_test)
    mape1=mape(predictions,y_test)
    return mse1,mape1,(end-start)
#********************************************************************************************************
   
#y=e^ax +c 
def second_method(x,y):
    start=time.time()
    X_train, X_test, y_train, y_test = train_test_split(
                             x, y, test_size=0.33, random_state=42)
    def cost_mse(list):
        mse = ((list[0]+np.exp(list[1]*-X_train)-y_train)**2).sum()/len(y_train)
        #print each iteration
        #print('mse value: {:0.4f} prodConst: {:.4f} expConst: {:.4f} const: {:.4f}'.format(mse,list[0],list[1],list[2]))  
        return mse 
    pkonst=minimize(cost_mse,x0=np.array([10,np.exp(1)]),method='Powell')
    end=time.time()
    mse2=mse((pkonst.x[0]+(np.exp(pkonst.x[1]*-X_test))),y_test)
    mape2=mape(pkonst.x[0]+(np.exp(pkonst.x[1]*-X_test)),y_test)
    return mse2,mape2,(end-start)
#********************************************************************************************************   

def third_method(x,y):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c  
    def mse(predicted,real):
        return ((real-predicted)**2).sum()/len(real)
    start=time.time()
    X_train, X_test, y_train, y_test = train_test_split(
                             x, y, test_size=0.33, random_state=42)
    popt, pcov = curve_fit(func, X_train, y_train,p0=[1,np.exp(1),5], maxfev=5000)
    end=time.time()
    mse3=mse(func(X_test, *popt),y_test)
    mape3=mape(func(X_test, *popt),y_test)
    return mse3,mape3,(end-start)
#********************************************************************************************************   

def fourth_method(x,y):
    def func(x, b, c):
        return np.exp(-b * x) + c  
    def mse(predicted,real):
        return ((real-predicted)**2).sum()/len(real)
    start=time.time()
    X_train, X_test, y_train, y_test = train_test_split(
                             x, y, test_size=0.33, random_state=42)
    popt, pcov = curve_fit(func, X_train, y_train,p0=[1,np.exp(1),5], maxfev=5000)
    end=time.time()
    mse3=mse(func(X_test, *popt),y_test)
    mape3=mape(func(X_test, *popt),y_test)
    return mse3,mape3,(end-start)
    
import pandas as pd
df=pd.read_csv("data.csv")

def process1(df,q):
    results1_df=pd.DataFrame(columns=["Cell","MSE","Passed"])
    total_mse_1=0
    total_mape_1=0
    total_time_passed_1=0
    failed_to_reach_optima_counter=0
    counter=0
    for cell in set(df.CELL.values):
        y=(df[cell==df.CELL].loc[:,"actualdltputbh"]).to_numpy()
        x=(df[cell==df.CELL].loc[:,"actualdlprbutilbh"]).to_numpy()
        try:
            mse1,mape1,time_passed1 = first_method(x,y)
            total_mse_1 +=mse1
            total_mape_1+=mape1
            total_time_passed_1 +=time_passed1
            results1_df=results1_df.append({"Cell":cell,"MSE":"{:.2f}".format(mse1),"MAPE":"{:.2f}".format(mape1),
                                            "Passed":"{:.6f}".format(time_passed1)},ignore_index=True)
        except Exception as e:
            print("p1:",e)
            failed_to_reach_optima_counter+=1
            results1_df=results1_df.append({"Cell":cell,"MSE":"-",
                                            "Passed":"-","MAPE":"-"},ignore_index=True)
        
        counter+=1
        print("Process 1:",counter)
            
    results1_df.to_csv(r"results_linear_with_split.csv")
    q.put(["Lineer Regression",total_mse_1,total_mse_1/(counter-failed_to_reach_optima_counter),
    total_mape_1,total_mape_1/(counter-failed_to_reach_optima_counter),total_time_passed_1,failed_to_reach_optima_counter])

def process2(df,q):
    results2_df=pd.DataFrame(columns=["Cell","MSE","Passed"])
    total_mse_2=0
    total_mape_2=0
    total_time_passed_2=0
    failed_to_reach_optima_counter=0
    counter=0
    for cell in set(df.CELL.values):
        y=(df[cell==df.CELL].loc[:,"actualdltputbh"]).to_numpy()
        x=(df[cell==df.CELL].loc[:,"actualdlprbutilbh"]).to_numpy()
    
        try:
            mse2,mape2,time_passed2 = second_method(x,y)
            total_mse_2 +=mse2
            total_mape_2+=mape2
            total_time_passed_2 +=time_passed2
            results2_df=results2_df.append({"Cell":cell,"MSE":"{:.2f}".format(mse2),"MAPE":"{:.2f}".format(mape2),
                                            "Passed":"{:.6f}".format(time_passed2)},ignore_index=True)
        except Exception as e:
            print("p2:",e)
            failed_to_reach_optima_counter+=1
            results2_df=results2_df.append({"Cell":cell,"MSE":"-",
                                            "Passed":"-","MAPE":"-"},ignore_index=True)
        
        counter+=1
        print("Process 2:",counter)
            
    results2_df.to_csv(r"results_powell_2D_with_split.csv") 
    q.put(["Powell 2D",total_mse_2,total_mse_2/(counter-failed_to_reach_optima_counter),
    total_mape_2,total_mape_2/(counter-failed_to_reach_optima_counter),total_time_passed_2,failed_to_reach_optima_counter] )  
    

def process3(df,q):
    results3_df=pd.DataFrame(columns=["Cell","MSE","Passed"])
    total_mse_3=0
    total_mape_3=0
    total_time_passed_3=0
    counter=0
    failed_to_reach_optima_counter=0
    for cell in set(df.CELL.values):
        y=(df[cell==df.CELL].loc[:,"actualdltputbh"]).to_numpy()
        x=(df[cell==df.CELL].loc[:,"actualdlprbutilbh"]).to_numpy()

        try:
            mse3,mape3,time_passed3 = third_method(x,y)
            total_mse_3 +=mse3
            total_mape_3+=mape3
            total_time_passed_3 +=time_passed3
            results3_df=results3_df.append({"Cell":cell,"MSE":"{:.2f}".format(mse3),"MAPE":"{:.2f}".format(mape3),
                                            "Passed":"{:.6f}".format(time_passed3)},ignore_index=True)
        except Exception as e:
            print("p3:",e)
            failed_to_reach_optima_counter+=1
            results3_df=results3_df.append({"Cell":cell,"MSE":"-",
                                            "Passed":"-","MAPE":"-"},ignore_index=True)
        counter+=1
        print("Process 3:",counter)
             
    results3_df.to_csv(r"results_curve_fit_3D_with_split.csv")
    q.put(["Curve Fit 3D",total_mse_3,total_mse_3/(counter-failed_to_reach_optima_counter),total_mape_3,
    total_mape_3/(counter-failed_to_reach_optima_counter),total_time_passed_3,failed_to_reach_optima_counter])

def process4(df,q):
    results4_df=pd.DataFrame(columns=["Cell","MSE","Passed"])
    total_mse_4=0
    total_mape_4=0
    total_time_passed_4=0
    counter=0
    failed_to_reach_optima_counter=0
    for cell in set(df.CELL.values):
        y=(df[cell==df.CELL].loc[:,"actualdltputbh"]).to_numpy()
        x=(df[cell==df.CELL].loc[:,"actualdlprbutilbh"]).to_numpy()

        try:
            mse4,mape4,time_passed4 = third_method(x,y)
            total_mse_4 +=mse4
            total_mape_4+=mape4
            total_time_passed_4 +=time_passed4
            results4_df=results4_df.append({"Cell":cell,"MSE":"{:.2f}".format(mse4),"MAPE":"{:.2f}".format(mape4),
                                            "Passed":"{:.6f}".format(time_passed4)},ignore_index=True)
        except Exception as e:
            print("p4:",e)
            failed_to_reach_optima_counter+=1
            results4_df=results4_df.append({"Cell":cell,"MSE":"-",
                                            "Passed":"-","MAPE":"-"},ignore_index=True)
        counter+=1
        print("Process 4:",counter)
             
    results4_df.to_csv(r"results_curve_fit_2D_with_split.csv")
    q.put(["Curve Fit 2D",total_mse_4,total_mse_4/(counter-failed_to_reach_optima_counter),total_mape_4,
    total_mape_4/(counter-failed_to_reach_optima_counter),total_time_passed_4,failed_to_reach_optima_counter])

def process11(df,q):
    from main import process1
    warnings.filterwarnings("ignore")
    process1(df,q)

def process22(df,q):
    from main import process2
    warnings.filterwarnings("ignore")
    process2(df,q)

def process33(df,q):
    warnings.filterwarnings("ignore")
    from main import process3
    process3(df,q)

def process44(df,q):
    warnings.filterwarnings("ignore")
    from main import process4
    process4(df,q)
     
from multiprocessing import Process,Queue
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    q = Queue()
    p1 = Process(target=process11, args=(df,q,))
    p2=Process(target=process22, args=(df,q,))
    p3=Process(target=process33, args=(df,q,)) 
    p4=Process(target=process44, args=(df,q,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p2.join()
    p1.join()
    p3.join()
    p4.join()
    
    results_df=pd.DataFrame(columns=["Method Name","Total MSE","Average MSE","Total MAPE","Average MAPE","Total Time Passed"])
    strlist=[]
    for i in range(4):
                strlist=(q.get())
                results_df=results_df.append({"Method Name":strlist[0],"Average MSE":"{:.2f}".format(strlist[2]),
                                    "Total MSE":"{:.2f}".format(strlist[1]),"Average MAPE":"{:.2f}".format(strlist[4]),
                                    "Total MAPE":"{:.2f}".format(strlist[3]),"Times Model Failed":strlist[6],
                                    "Total Time Passed":strlist[5]},ignore_index=True)
    results_df.to_csv(r"result_with_split.csv")
    a=input("press a key to exit")