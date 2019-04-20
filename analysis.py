import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.api import OLS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#Load Data

dataset_doctors_us = pd.read_csv("US_PHYSICIANS_PER_1000.csv")

dataset_marriage_us = pd.read_csv("US_AGE_MARRIAGE_WOMEN.csv")

dataset_retirement_us = pd.read_csv("US_AGE_RETIREMENT.csv")

dataset_mortality_us = pd.read_csv("US_MORTALITY.csv")

dataset_fertility_us = pd.read_csv("US_FERTILITY.csv")

dataset_hoursworked_us = pd.read_csv("US_HOURS_WORKED.csv")

dataset_alcohol_us = pd.read_csv("US_ALCOHOL.csv")

dataset_bmi_us = pd.read_csv("US_BMI.csv")

dataset_mortality_br = pd.read_csv("BR_MORTALITY.csv")

dataset_lifeexpectancy_us = pd.read_csv("US_LIFE_EXPECTANCY.csv")

dataset_developing_independent = pd.read_csv("DVLP_FULL.csv")

dataset_developing_lifeexpectancy = pd.read_csv("DVLP_LIFEEXPECTANCY.csv")

dataset_developing_fertility = pd.read_csv("DVLP_FERTILITY.csv")


#Delete empty columns

dataset_marriage_us.drop(dataset_marriage_us.columns[dataset_marriage_us.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
dataset_retirement_us.drop(dataset_retirement_us.columns[dataset_retirement_us.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
dataset_mortality_us.drop(dataset_mortality_us.columns[dataset_mortality_us.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
dataset_fertility_us.drop(dataset_fertility_us.columns[dataset_fertility_us.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)



dataset_us = pd.concat([dataset_marriage_us,dataset_retirement_us, dataset_alcohol_us, dataset_hoursworked_us,dataset_bmi_us,dataset_doctors_us], axis = 1)

X = dataset_developing_independent
Y = dataset_developing_fertility

X = sm.add_constant(X)

print(OLS(Y,X).fit().summary())


