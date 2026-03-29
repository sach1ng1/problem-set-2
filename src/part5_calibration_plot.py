'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()
    
df_arrests_test=pd.read_csv("data/df_arrests_test.csv")
# Calibration curve for logistic regression model
calibration_plot(df_arrests_test["y"],df_arrests_test["pred_lr"], n_bins=5)
# Calibration curve for Decision tree model
calibration_plot(df_arrests_test["y"], df_arrests_test["pred_dt"], n_bins=5)

print("Which model is more calibrated?")
print("From what i can observe both models have the same calibration so one model does not seem to be more calibrated then the other.")

top_lr_predicted_risk=df_arrests_test.nlargest(50,"pred_lr")
ppv_logistic_regression=top_lr_predicted_risk["y"].mean()

top_dt_predicted_risk=df_arrests_test.nlargest(50,"pred_dt")
ppv_decision_tree=top_dt_predicted_risk["y"].mean()

print(f"Logistic PPV: {ppv_logistic_regression}")
print(f"Decision Tree PPV:{ppv_decision_tree}")

auc_logistic_regression=roc_auc_score(df_arrests_test["y"], df_arrests_test["pred_lr"])
auc_decision_tree= roc_auc_score(df_arrests_test["y"], df_arrests_test["pred_dt"])

print(f"Logistic AUC: {auc_logistic_regression}")
print(f"Decision Tree AUC: {auc_decision_tree}")

print("Yes, in both metrics it is shown that one model is more accurate than the other.")


    