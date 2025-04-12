import numpy as np
import pandas as pd

a=[np.array([1,2,3,4,5,6,7,8,9]),np.array([4,5]),np.array([6])]

dataframe=pd.DataFrame.from_records(a)
dataframe.to_csv("test.csv")

dataframe=pd.read_csv("test.csv")
new_list=dataframe.loc[1].to_list()
cleanedList = [x for x in new_list if str(x) != 'nan']
print(np.array(cleanedList))