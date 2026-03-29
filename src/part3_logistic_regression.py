'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr

def logistic_regression(df_arrests):
    """ Initiates and runs the logistic regression model from the merged arrest data

    Parameter:
        df_arrests: dataframe
            The merged and preprocessed arrest event and individual description data
    
    Returns:
        df_arrests_train:
            dataframe with the train data for arrests
        df_arrests_test:
            dataframe with the test data for arrests
    """    
    df_arrests_train,df_arrests_test=train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify= df_arrests["y"], random_state=42)


    features=["current_charge_felony", "num_fel_arrests_last_year"] 

    X_train=df_arrests_train[features]
    y_train=df_arrests_train["y"]

    X_test=df_arrests_test[features]
    y_test=df_arrests_test["y"]

    param_grid={"C":[0.05, 0.5, 1]}

    lr_model=lr()
    gs_cv=GridSearchCV(lr_model, param_grid=param_grid, cv=5)
    gs_cv.fit(X_train, y_train)

    optimal_c=gs_cv.best_params_["C"]
    print(f"What was the optimal value for C? {optimal_c}")

    if optimal_c == min(param_grid["C"]):
        regularization= "most regularization"
    elif optimal_c == max(param_grid["C"]):
        regularization="least regularization"
    else:
        regularization="middle regularization"
        
    print(f"Did it have the most or least regularization? Or in the middle? {regularization}")

    df_arrests_test["pred_lr"]=gs_cv.predict(X_test)
    
    return df_arrests_train, df_arrests_test

    
    
    