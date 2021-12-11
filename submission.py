### IMPORT YOUR FORCAST FUNCTION

### IMPORT YOUR DECISION FUNCTION

from config import TRAIN_DATASET_PATH
### SET SUBMISSION START
EXAM_FILE_PATH      = TRAIN_DATASET_PATH
RESULT_FILE_NAME    = "submission_train"
PRED_FUNCTION       = any
SKIP_DECISION       = True
### SET SUBMISSION END

import numpy as np
import pandas as pd
from datetime import datetime
df_exam = pd.read_csv(EXAM_FILE_PATH, index_col=0)

def submission(exam_path:str, func, result_path="./headcount.csv", skip_decision=False)->bool:
    """ function for make submission file (*.csv)
    """
    df_exam = pd.read_csv(exam_path, index_col=0)
    n_exam = len(df_exam)
    result_data = np.zeros((n_exam,2)) # Column: [avg gas production of 6 month, with or without selection]
    # Forecasting Head Counts
    for num in range(n_exam):
        result_data[num][0] = func(df_exam.iloc[num])
    df_exam.to_csv(result_path)
    return True

if __name__ == "__main__":
    headcount_file_path = "./{0}_headCount_{1}.csv".format(
                                            RESULT_FILE_NAME,
                                            datetime.now().strftime("%Y%m%d%H%M%S"))
    submission(
            EXAM_FILE_PATH,
            PRED_FUNCTION,
            product_result_path=headcount_file_path,
            skip_decision=SKIP_DECISION)
    print("[SUBMISSION] Result file for Head Count: {0}".format(headcount_file_path))