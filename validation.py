### SET FILE PATH
TRUE_FILE_PATH = "./forecast/answer.csv"
### SET FILE PATH END
from config import TRAIN_DATASET_PATH
from submission import submission
### IMPORT YOUR FORCAST FUNCTION
### IMPORT YOUR DECISION FUNCTION
### SET SUBMISSION START
EXAM_FILE_PATH      = TRAIN_DATASET_PATH
RESULT_FILE_NAME    = "submission_train"
PRED_FUNCTION       = any
SKIP_DECISION       = True
### SET SUBMISSION END

from datetime import datetime
import numpy as np
def sMAPE(A, F)->float:
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def sMAPE_static(A, F)->float:
    return sMAPE(A[:48], F[:48])

def sMAPE_serial(A, F)->float:
    return sMAPE(A[48:], F[48:])

def compare_two_csv_files(file1:str, file2:str):
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    actual_value = np.loadtxt(file1,delimiter=',')[:,0]
    fake_value = np.loadtxt(file2,delimiter=',')[:,0]
    print("***** Calculate sMAPE *****")
    print("- Time Stamp  : {0}".format(date_time))
    print("- Actual file : {0}".format(file1))
    print("- Fake   file : {0}".format(file2))
    print("- sMAPE       : {0} %".format(sMAPE(actual_value,fake_value)))
    print("--    Static  : {0} %".format(sMAPE_static(actual_value,fake_value)))
    print("--    Serial  : {0} %".format(sMAPE_serial(actual_value,fake_value)))
    print("***** Calculate sMAPE END *****")

if __name__ == "__main__":
    headcount_file_path = "./{0}_headCount_{1}.csv".format(
                                            RESULT_FILE_NAME,
                                            datetime.now().strftime("%Y%m%d%H%M%S"))
    submission(
            EXAM_FILE_PATH,
            PRED_FUNCTION,
            product_result_path=headcount_file_path,
            skip_decision=SKIP_DECISION)
    compare_two_csv_files(TRUE_FILE_PATH,headcount_file_path)
