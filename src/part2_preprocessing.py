'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import numpy as np

def preprocess_df():
    """Preprocesses data to identify individuals with felony charges to prepare for modeling

    Returns:
        df_arrests:
            merged and preprocessed dataframe of arrests_events_raw and pred_universe_raw dataframes containing arrest event and individual descriptions
            
    """    

    arrest_events_df=pd.read_csv("data/arrest_events_raw.csv")
    pred_universe_df=pd.read_csv("data/pred_universe_raw.csv")
    arrest_events_df["arrest_date_event"]=pd.to_datetime(arrest_events_df["arrest_date_event"])
    pred_universe_df["arrest_date_univ"]=pd.to_datetime(pred_universe_df["arrest_date_univ"])
    df_arrests=pd.merge(arrest_events_df, pred_universe_df, how="outer", on="person_id")

    #print(df_arrests.head())
    #print(df_arrests.info())

    y_felony=[]

    for index, arrest in df_arrests.iterrows():
        arrested_person= arrest["person_id"]
        arrest_date= arrest["arrest_date_univ"]
        
        rearrested= df_arrests[(df_arrests["person_id"]==arrested_person) & (df_arrests["arrest_date_event"]> arrest_date)&
        (df_arrests["arrest_date_event"]<=arrest_date + pd.DateOffset(years=1)) & (df_arrests["charge_degree"]=="felony")]
        
        if len(rearrested)>0:
            y_felony.append(1)
        else:
            y_felony.append(0)
            
    #print(y_felony)     
            
    df_arrests["y"]=y_felony
        
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {df_arrests['y'].mean()}")

    felony_charges=[]  
        
    for index, arrest in df_arrests.iterrows():
        if arrest["charge_degree"]== "felony":
            felony_charges.append(1)
        else:
            felony_charges.append(0)
    #print(current_arrests)
    df_arrests["current_charge_felony"] =felony_charges

    print(f"What share of current charges are felonies? {df_arrests["current_charge_felony"].mean()}")

    past_arrests=[]

    for index, arrest in df_arrests.iterrows():
        arrested_person= arrest["person_id"]
        arrest_date= arrest["arrest_date_univ"]
        
        past_felonies=df_arrests[(df_arrests["person_id"]==arrested_person)& (df_arrests["arrest_date_event"]<arrest_date)&
        (df_arrests["arrest_date_event"]>=arrest_date-pd.DateOffset(years=1))& (df_arrests["charge_degree"]=="felony")]
        past_arrests.append(len(past_felonies))

    df_arrests["num_fel_arrests_last_year"]= past_arrests

    print(f"What is the average number of felony arrests in the last year? {df_arrests["num_fel_arrests_last_year"].mean()}")

    print(pred_universe_df.head())
    return df_arrests






    
    
    





    
    
    



