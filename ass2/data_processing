import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('5220001_Annual_Gross_State_Product_All_States.xls', sheet_name = ['Data1'])
data_2 = pd.read_excel('3101052.xls', sheet_name=['Data1', 'Data2'])

victoria_economy = data['Data1']
victoria_population1 = data_2['Data1']
victoria_population2 = data_2['Data2']

output_1 = victoria_economy.iloc[:, [0, 2, 11, 20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110, 119]] #Victoria's data
output_1.to_csv("economy.csv", index = False)
