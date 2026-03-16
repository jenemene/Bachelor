import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def statisticalplot(times,n_list,repeats):
    #calculation of mean, standard eviation, and standard error
    mean_times = np.mean(times,axis=1)
    std_dev = np.std(times,axis=1)
    std_err = std_dev/np.sqrt(repeats) #std errors used for error

    #scatter plot with error bars
    plt.errorbar(n_list,mean_times,yerr=std_err,color="k",fmt="o")

    #linear polyfit
    coeffs = np.polyfit(n_list,mean_times,1)
    y_linreg = coeffs[0]*n_list + coeffs[1]
    plt.plot(n_list,y_linreg,"r--")

    #plotting settings
    plt.grid()
    plt.xlabel("No. of bodies")
    plt.ylabel("Mean simulation times")
    plt.title("Mean simulation time vs no of bodies (repitions = 10)")
    plt.show()

#data for 10 bodies. This could be implemented as automatic extraction if one wanted. 
repeats = 10
n_bodies = 10
n_list = np.arange(1,n_bodies+1,1)

#reading data generated
df = pd.read_csv("pendulum_benchmark.csv")
simu_times = df["duration"].to_numpy()
times = simu_times.reshape(n_bodies,repeats) #rows = no of bodies, columns = repitions


#plotting and data analyis

statisticalplot(times, n_list, repeats)