# 경사하강법(해석편미분)

import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'library'))
    from common import mean_squares_Error
except ImportError:
    print('Library Module Can Not Found')
import numpy as np


def analytic_gradient(x, data_training):
    dx = np.zeros_like(x)

    n = len(data_training[0])
    error = x[0] * data_training[0] + x[1] - data_training[1] # e = ax+b - y (여기서 x와 y는 학습 데이터)
    # 오차(error)는 각 학습 데이터 times와 scores에 따라 줄었다 늘었다를 반복하며 결과적으로는
    # 최대한 직선 그래프와 오차를 최소화하는 방향으로 나아간다.
    dx[0] = (2 / n) * np.sum(error * data_training[0])  # 손실함수의 기울기 구하기(a)
    dx[1] = (2 / n) * np.sum(error) # 손실함수의 절편 구하기(b)
    # 위 손실함수의 기울기와 절편을 구하는 공식은 수학적 전공 지식이 어느 정도 필요한 부분으로
    # 실제 딥러닝에서는 해석 미분이 아닌 수치 미분을 시행하므로 위와 같은 예시가 있다는 정도만
    # 알고 넘어가도 된다. 목표는 a와 b가 0에 수렴하는 손실함수를 구하고 그때의 직선 그래프를
    # 구하는 것이다.
    return dx

# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

# 경사하강법
x = np.array([0., 0.]) # x[0]이 선형회귀의 기울기, x[1]이 선형회귀의 절편값이다. [0., 0.]으로 선언한 건 초기값으로 임의 지정한 것.
lr = 0.01
epoch = 5000
for i in range(epoch):
    gradient = analytic_gradient(x, (np.array(times), np.array(scores)))
    print(f'epoch={i + 1}, gradient={gradient}, x={x}')

    x -= lr * gradient

a, b = tuple(x)
print(f'직선 y = {a}x + {b}')
print(f'오차(평균제곱오차):{mean_squares_Error(np.array([a, b]), (times, scores))}')
