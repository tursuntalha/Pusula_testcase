


## How to Run the Project

1. **Clone the Repository**

2. **Navigate to the Project Directory**

   cd Pusula_Talha_Tursun

3. **Install Libraries**
 
   pip install pandas numpy scikit-learn openpyxl

4. **Open Jupyter Notebook**

   jupyter notebook





The dataset path is set according to Jupyter Notebook



## Data Analysis

In this section, I identify anomalies in the dataset using Matplotlib and Seaborn.
I determine how many missing values exist in each column.
I check the data types of the columns.
I find anomalies in columns such as User_id, Nationality, My_Diseases, and datetime64.


## Data Preprocessing

Here, I begin to correct the anomalies identified during the Data Analysis phase.
I adjusted the ID, nationality, and gender structures to make them suitable for the dataset.
I generated new values from date-type data that are more useful for modeling.
I expanded the My_Diseases columns to make them cleaner and more effective for modeling.
I filled in the missing values in the My_Diseases columns based on their interrelationships.
For the remaining data, numerical values were filled using KNN, while categorical values were filled using SimpleImputer.
The resulting values were converted to numerical representations using LabelEncoder.
Finally, the dataset was prepared for modeling using StandardScaler.








## You can find a more detailed and Turkish project description in the documentation file in the repository. :)






