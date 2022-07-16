from logging import exception
import time
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from multiprocessing import Process,Queue


def process(method_fucntion,method_name,df,q):
    import main 
    warnings.filterwarnings("ignore")
    results1_df=pd.DataFrame(columns=["Cell","MSE","Passed"])
    total_mse_1=0
    total_mape_1=0
    total_time_passed_1=0
    failed_to_reach_optima_counter=0
    counter=len(set(df.CELL.values))
    for cell in set(df.CELL.values):
        y=(df[cell==df.CELL].loc[:,"actualdltputbh"]).to_numpy()
        x=(df[cell==df.CELL].loc[:,"actualdlprbutilbh"]).to_numpy()/100
        try:
            mse1,mape1,time_passed1 = method_fucntion(x,y)
            total_mse_1 +=mse1
            total_mape_1+=mape1
            total_time_passed_1 +=time_passed1
            results1_df=results1_df.append({"Cell":cell,"MSE":"{:.2f}".format(mse1),"MAPE":"{:.2f}".format(mape1),
                                            "Passed":"{:.6f}".format(time_passed1)},ignore_index=True)
        except Exception as e:
            print("Exception Message for Process of "+method_name,":\n\t",e)
            failed_to_reach_optima_counter+=1
            results1_df=results1_df.append({"Cell":cell,"MSE":"-",
                                            "Passed":"-","MAPE":"-"},ignore_index=True)
        
 
    csv_name="results_"+method_name+".csv" 
    results1_df=results1_df.sort_values(by=['Cell'])     
    results1_df.to_csv(csv_name)
    results1_df=results1_df[results1_df.MAPE!="-"]
    q.put([method_name,total_mse_1,total_mse_1/(counter-failed_to_reach_optima_counter),
    total_mape_1,total_mape_1/(counter-failed_to_reach_optima_counter),total_time_passed_1,failed_to_reach_optima_counter,
    results1_df.MAPE.astype(float).std(),results1_df.MSE.astype(float).std()])

def mse(predicted,real):
    return mean_squared_error(predicted,real)

def mape(pred,actual): 
    return mean_absolute_error(pred,actual)
#********************************************************************************************************
 
def method_1(x,y):
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
    predictions=np.squeeze(np.exp((predictions.reshape(1,-1))))
    y_test=np.squeeze(np.exp(y_test.reshape(1,-1)))
    end=time.time()
    mse1=mse(predictions,y_test)
    mape1=mape(predictions,y_test)
    return mse1,mape1,(end-start)
#********************************************************************************************************  
#y=1,0335^(-x+b) +c
def method_2(x,y):
    start=time.time()
    X_train, X_test, y_train, y_test = train_test_split(
                             x, y, test_size=0.33, random_state=42)
    def cost_mse(list):
        mse = ((((list[0]+(26.98**(-X_train+list[1])))-y_train)**2)).sum()/len(y_train)
        return mse 
    pkonst=minimize(cost_mse,x0=np.array([1,1.28]),method='Powell')
    end=time.time()
    mse2=mse((pkonst.x[0]+(26.98**(-X_test+pkonst.x[1]))),y_test)
    mape2=mape((pkonst.x[0]+(26.98*(-X_test+pkonst.x[1]))),y_test)
    return mse2,mape2,(end-start)
#********************************************************************************************************   

def method_3(x,y):
    def func(x, a, b, c):
        return a * (b ** -x) + c  
    start=time.time()
    X_train, X_test, y_train, y_test = train_test_split(
                             x, y, test_size=0.33, random_state=42)
    popt, pcov = curve_fit(func, X_train, y_train,p0=[68,26.98,1],bounds=([0,1,0], [670, 1200000, 30]), maxfev=5000)
    end=time.time()
    mse3=mse(func(X_test, *popt),y_test)
    mape3=mape(func(X_test, *popt),y_test)
    return mse3,mape3,(end-start)
#********************************************************************************************************   

def method_4(x,y):
    def func(x, b, c):
        return (26.98 **(-x+b) ) + c  
    start=time.time()
    X_train, X_test, y_train, y_test = train_test_split(
                             x, y, test_size=0.33, random_state=42)
    popt, pcov = curve_fit(func, X_train, y_train,p0=[1.28,1])
    end=time.time()
    mse3=mse(func(X_test, *popt),y_test)
    mape3=mape(func(X_test, *popt),y_test)
    return mse3,mape3,(end-start)
#*******************************************************************************************************

if __name__ == '__main__':
    df=pd.read_csv("data.csv")
    warnings.filterwarnings("ignore", category=FutureWarning)
    q = Queue()
    p1 = Process(target=process, args=(method_1,"Linear_Regression",df,q,))
    p2=Process(target=process, args=(method_4,"Curve_Fit_2D",df,q,))
    p3=Process(target=process, args=(method_3,"Curve_Fit_3D",df,q,)) 
    p4=Process(target=process, args=(method_2,"Powell_2D",df,q,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join() 
    p4.join()
    
    
    results_df=pd.DataFrame(columns=["Method Name","Total MSE","Average MSE","SD MSE","Total MAPE","Average MAPE",
    "SD MAPE","Total Time Passed"])
    strlist=[]
    for i in range(4):
        
                strlist=(q.get())
                results_df=results_df.append({"Method Name":strlist[0],"Total MSE":"{:.2f}".format(strlist[1]),
                                    "Average MSE":"{:.2f}".format(strlist[2]),
                                    "SD MSE":"{:.2f}".format(strlist[8]),"Total MAPE":"{:.2f}".format(strlist[3]),
                                    "Average MAPE":"{:.2f}".format(strlist[4]),"SD MAPE":"{:.2f}".format(strlist[7]),
                                    "Times Model Failed":strlist[6],
                                    "Total Time Passed":"{:.3f}".format(strlist[5])},ignore_index=True)
                
    results_df.to_csv(r"result_with_split.csv")

