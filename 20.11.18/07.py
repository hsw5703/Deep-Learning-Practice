# 다중선형회귀(수치미분)
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'library'))
    from common import gradient_descent
except ImportError:
    print('Library Module Can Not Found')
import numpy as np


def mean_squares_Error(x, data_training):
    data_x0, data_x1, data_y = data_training

    s = 0
    for i in range(len(data_x0)):
        data_y_hat = x[0] * data_x0[i] + x[1] * data_x1[i] + x[2] # y(예측값) = a0x0+a1x1+b
        s += ((data_y_hat - data_y[i]) ** 2) # 예측값과 학습데이터의 차이의 제곱
    e = s / len(data_x0)

    return e


# data 종속 변수가 두 개로 늘었을 경우.
times = [2, 4, 6, 8]
ptimes = [0, 4, 2, 3]
scores = [81, 93, 91, 97]

# 경사하강법
result = gradient_descent(mean_squares_Error, np.array([0., 0., 0.]), epoch=5000, data_training=(times, ptimes, scores))
# 여기서 사용된 mean_squares_Error는 위에 선언된 함수이다. common.py에 있는 함수 X.
print(result)

# ==============================================================================================


# predict(inference)
x1_p = 2
x2_p = 2
y_p = result[0] * x1_p + result[1] * x2_p + result[2]
print(y_p)