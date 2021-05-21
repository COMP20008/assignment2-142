import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Run this to get Scatterplot_Population_GSP, year_gsp, year_population graph

# pd.set_option('display.max_columns', None)
import data_processing
victoria_economy = pd.read_csv('economy.csv',encoding = 'ISO-8859-1')
## Data cleaning for population data
population = pd.read_excel('3101052.xls', sheet_name = ['Data1', 'Data2'])

victoria_population1 = population['Data1']
victoria_population2 = population['Data2']
male = list(victoria_population1.iloc[:, 1:102]) #Get all male columns
female = list(victoria_population1.iloc[:, 102:203]) #Get all female columns
person = list(victoria_population1.iloc[:, 203:251]) #Get all person columns
person2 = list(victoria_population2.iloc[:, 1:54])

victoria_male = victoria_population1[male].iloc[9:].astype(int) #Skip useless header
victoria_female = victoria_population1[female].iloc[9:].astype(int)
victoria_person1 = victoria_population1[person].iloc[9:].astype(int)
victoria_person2 = victoria_population2[person2].iloc[9:].astype(int)
victoria_person = pd.merge(victoria_person1, victoria_person2, left_index=True, right_index=True) #Merge both person dataframe

victoria_male_sum = victoria_male.sum(axis = 1) #Calculate sum for population of all ages
victoria_female_sum = victoria_female.sum(axis = 1)
victoria_person_sum = victoria_person.sum(axis = 1)

male_result = victoria_male_sum.iloc[19:] #Only obtain data in june starting from 1990
female_result = victoria_female_sum.iloc[19:]
person_result = victoria_person_sum.iloc[19:]

# Data pre-processing for GSP data
economy_data = victoria_economy.iloc[9:] #Skip useless header

# Convert data to int and float for processing
economy_data_int = economy_data.iloc[:, 1:15].astype(float)
year = pd.to_datetime(economy_data.iloc[:, 0]).dt.year #Only year for simplification

# Create list of male population percentage change
first = 1
male_list = []
for i in male_result:
    if first:
        new = i
    else:
        old = new
        new = i
        percentage_change = (new-old)/old
        male_list.append(percentage_change)
    first = 0
# male_list.append(0)
# Create list of female population percentage change
first = 1
female_list = []
for i in female_result:
    if first:
        new = i
    else:
        old = new
        new = i
        percentage_change = (new-old)/old
        female_list.append(percentage_change)
    first = 0
# female_list.append(0)
#Create list of person population percentage change
first = 1
person_list = []
for i in person_result:
    if first:
        new = i
    else:
        old = new
        new = i
        percentage_change = (new-old)/old
        person_list.append(percentage_change)
    first = 0
# person_list.append(0)
#Add colume: GSP chain volume measures percentage change
first = 1
list = []
for j in economy_data_int['Victoria ;  Gross state product: Chain volume measures ;']:
    if first:
        new = j
    else:
        old = new
        new = j
        percentage_change = (new-old)/old
        list.append(percentage_change)
    first = 0
list.append(0) #No percentage change in last row

economy_data_int['Victoria ;  Gross state product: Chain volume measures ; Percentage change ;'] = list
x = person_result
y = economy_data_int['Victoria ;  Gross state product: Chain volume measures ;']
print(np.corrcoef(x, y))
#Year vs population
m, b = np.polyfit(year, person_result, 1)
plt.figure()
plt.scatter(year, person_result, color='black', label= "Person")
plt.scatter(year, male_result, color='blue', label= "Male")
plt.scatter(year, female_result, color='red', label= "Female")
plt.plot(year, m*year + b)
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title("Scatterplot of year against population in Victoria")
plt.savefig("Year_Population")

#Year vs population - percentage change
#Note: Ignore last data (year: 2020) since percantage change cant be calculated in the last year
#      Percentage change in year 2019 is too similar for both genders, so it looks like there's only one data point
m, b = np.polyfit(year.iloc[0:30], person_list, 1)
plt.figure()
plt.scatter(year.iloc[0:30], person_list, color='black', label= "Person")
plt.scatter(year.iloc[0:30], male_list, color='blue', label= "Male")
plt.scatter(year.iloc[0:30], female_list, color='red', label= "Female")
plt.plot(year, m*year + b)
plt.xlabel("Year")
plt.ylabel("Population - Percentage change")
plt.legend()
plt.title("Scatterplot of year against population in Victoria - Percentage change")
plt.savefig("Year_Population_Percentage")

#Year vs gsp chain volume measures
m, b = np.polyfit(year, economy_data_int['Victoria ;  Gross state product: Chain volume measures ;'], 1)
plt.figure()
plt.scatter(year, economy_data_int['Victoria ;  Gross state product: Chain volume measures ;'], color='black')
plt.plot(year, m*year + b)
plt.xlabel("Year")
plt.ylabel("GSP: Chine volume measures")
plt.title("Scatterplot of year against GSP in Victoria")
plt.savefig("Year_GSP")

#Year vs gsp chain volume measures - percentage change

m, b = np.polyfit(year.iloc[0:30], economy_data_int['Victoria ;  Gross state product: Chain volume measures ; Percentage change ;'].iloc[0:30], 1)
plt.figure()
plt.scatter(year.iloc[0:30], economy_data_int['Victoria ;  Gross state product: Chain volume measures ; Percentage change ;'].iloc[0:30], color='black')
plt.plot(year, m*year + b)
plt.xlabel("Year")
plt.ylabel("GSP: Chine volume measures - Percentage change")
plt.title("Scatterplot of year against GSP in Victoria - Percentage change")
plt.savefig("Year_GSP_Percentage")

#Population vs gsp chain volume measures
plt.figure()
plt.scatter(person_result, economy_data_int['Victoria ;  Gross state product: Chain volume measures ;'], color='black', label= "Person")
plt.xlabel("Population")
plt.ylabel("GSP: Chain volume measures")
plt.title("Scatterplot of population against GSP: chain volume measures in Victoria")
plt.savefig("Scatterplot_Population_GSP")
