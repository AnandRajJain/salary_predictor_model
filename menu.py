import pandas as pd

sal = pd.read_csv('salary_prediction.csv')

x = sal['YearsExperience'].values.reshape(30,1)

y = sal['Salary']
from sklearn.linear_model import LinearRegression
salmodel = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
salmodel.fit(X_train , y_train)
def sal_predict():
    print("                                                                                                        ")
    exp_years = float(input("Enter your experience in years:"))
    print("                                                                                                        ")
    sal_predict = salmodel.predict([[exp_years]])
    print("                                                                                                        ")
    print("The salary for the submitted years of experience is : ",sal_predict)
    
print("----------------------------------Welcome to Salary Predictor Model-------------------------------------")
print("                                                                                                        ")
print("                                                                                                        ")
print("To Predict the Salary Please Type 'YES' and Type 'NO' to exit")
while True:
    #choice = input("\n Please type  'Yes' to continue to prediction or type 'No' to end  the prediction: ")
    choice = input("\n Please type your input : ");
    
    if choice == "YES":
        sal_predict()
        print("                                                                                                        ")
    elif choice == "NO":
        print("                                                                                                        ")
        print("Thanks for using our Model")
        break
    else:
        print("\n Please re enter the input")
    print("To continue Predicting the Salary Please Type 'YES' and to Stop Predicting Please type 'NO'")
