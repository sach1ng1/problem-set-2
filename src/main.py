'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
from part1_etl import savedata
from part2_preprocessing import preprocess_df
from part3_logistic_regression import logistic_regression
from part4_decision_tree import decision_tree
from part5_calibration_plot import calibration_plot


# Call functions / instanciate objects from the .py files
def main():
    """ Main function calls the functions from Parts 1 through 5 .py files
    """    

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    pred_universe_raw, arrest_events_raw= savedata()

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests=preprocess_df()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test=logistic_regression(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree
    decision_tree(df_arrests_train, df_arrests_test)
    # PART 5: Call functions/instanciate objects from calibration_plot
    df_arrests_test=pd.read_csv("data/df_arrests_test.csv")
    calibration_plot(df_arrests_test["y"], df_arrests_test["pred_lr"], n_bins=5)
    calibration_plot(df_arrests_test["y"], df_arrests_test["pred_dt"], n_bins=5)

if __name__ == "__main__":
    main()