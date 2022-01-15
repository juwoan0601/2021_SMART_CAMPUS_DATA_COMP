import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def k_ford_cross_validation(model, x, y):

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate

    from sklearn.model_selection import KFold

    from collections import Counter

    kfold = KFold(n_splits = 5, shuffle = True, random_state = 10)
    # scikit-learn 0.22 버전부터 기본적으로 5-겹 교차 검증으로 바뀌었다.
    basic_scores = cross_val_score(model, x, y)
    # 물론 cv 매개변수를 이용하여 k겹의 k를 변경가능하다.
    # 하지만, 교차검증에서는 대게 5겹 교차 검증을 자주 사용한다.
    kfold_scores = cross_val_score(model, x, y, cv = kfold)

    res = cross_validate(model, x, y, cv = 5, return_train_score=True)

    df = pd.DataFrame(res, index = ['case1', 'case2', 'case3', 'case4', 'case5'])
    #print(df)
    print('기본 교차 검증 점수 : ', basic_scores)
    print("기본 교차 검증 평균 점수 : {:.2f}".format(basic_scores.mean()))
    print('5겹 교차 검증 점수 (분할기 사용) : ', kfold_scores)
    print("5겹 교차 검증 평균 점수 (분할기 사용) : {:.2f}".format(kfold_scores.mean()))


def stratified_k_ford_cross_validation(model, x, y):

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold

    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 10)
    # scikit-learn 0.22 버전부터 기본적으로 5-겹 교차 검증으로 바뀌었다.
    basic_scores = cross_val_score(model, x, y)
    # 물론 cv 매개변수를 이용하여 k겹의 k를 변경가능하다.
    # 하지만, 교차검증에서는 대게 5겹 교차 검증을 자주 사용한다.
    kfold_scores = cross_val_score(model, x, y, cv = kfold)

    print('기본 교차 검증 점수 : ', basic_scores)
    print("기본 교차 검증 평균 점수 : {:.2f}".format(basic_scores.mean()))
    print('5겹 교차 검증 점수 (분할기 사용) : ', kfold_scores)
    print("5겹 교차 검증 평균 점수 (분할기 사용) : {:.2f}".format(kfold_scores.mean()))


def loocv(model, x, y):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    scores = cross_val_score(model, x, y, cv=loo)

    # 실제로 실행할 때 다른 교차 검증에 비해 오래 걸렸다.
    print("교차 검증 분할 횟수 : ", len(scores))
    print("평균 정확도 : {:.2f}".format(scores.mean()))

# def a():
#     from sklearn.model_selection import cross_val_score
#     from sklearn.model_selection import ShuffleSplit
#     from sklearn.model_selection import RepeatedStratifiedKFold
