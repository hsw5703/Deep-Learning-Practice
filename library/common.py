import numpy as np
from inspect import signature

def numerical_gradient(f, x, data_training=None): # 기울기
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i] # 수치편미분 변수가 여러 개 일 때.

        x[i] = tmp + h
        h1 = f(x) if len(signature(f).parameters) == 1 else f(x, data_training) # f(x) 넣으면 벡터라서 제곱하고 둘을 더해준다.
        x[i] = tmp - h
        h2 = f(x) if len(signature(f).parameters) == 1 else f(x, data_training)
        gradient[i] = (h1 - h2) / (2 * h)
        x[i] = tmp

        # h1 = f(x[i] + h) # 변수가 하나인 경우엔 이렇게 간단히 표현가능하지만 둘 이상의 경우엔 위처럼 해주어야 한다.
        #                  # 예를 들어, y = x0^2 + x1^2 이라는 다변수 함수를 편미분하려 할 때,
        #                  # x[i] + h가 x0에 들어갈지 x1에 들어갈지 결정할 수 없게 된다.
        #                  # 고로 위와 같이 해주어야 각각의 변수에 값을 대입할 수 있다.
        # h2 = f(x[i] - h) # 수치미분
        # gradient[i] = (h1 - h2) / (2 * h)

    return gradient


def gradient_descent(f, x, lr=0.01, epoch=100, data_training=None):  # 경사하강법
    for i in range(epoch):
        gradient = numerical_gradient(f, x, data_training)

        print(f'epoch={i + 1}, gradient={gradient}, x={x}')

        x -= lr * gradient
    return x


def method_least_squares(x, y):  # 최소제곱법
    mx = sum(x) / len(x)
    my = sum(y) / len(y)

    mls_a = sum([(i - mx) * (j - my) for i, j in zip(x, y)]) / sum([(i - mx) ** 2 for i in x])
    mls_b = my - (mx * mls_a)

    return mls_a, mls_b


def mean_squares_Error(x, data_training):  # 평균제곱오차(MSE, Mean Squares Error)
    # s = 0
    # for i in range(len(data_x)):
    #     data_y_hat = x[0] * data_x[i] + x[1]
    #     s += (data_y_hat - data_y[i]) ** 2
    # e = s / len(data_x)

    data_x, data_y = data_training
    data_y_hat = [x[0] * dx + x[1] for dx in data_x] # 최소제곱법에서 구한 ax+b에 주어진 data data_x[]를 대입.
    e = np.mean([(dyh - dy) ** 2 for dyh, dy in zip(data_y_hat, data_y)])
    return e

# --------------------------------------------------------------------------------------------------------
# def numerical_gradient_training(f, x, data_training):
#     h = 1e-4
#     gradient = np.zeros_like(x)
#
#     for i in range(x.size):
#         tmp = x[i]
#
#         x[i] = tmp + h
#         h1 = f(x, *data_training)
#         x[i] = tmp - h
#         h2 = f(x, *data_training)
#         gradient[i] = (h1 - h2) / (2 * h)
#
#         x[i] = tmp
#
#     return gradient
#
# def gradient_descent_linear_regression(f, x, lr=0.01, epoch = 100, data_training = None) : # 경사하강법 - 선형 회귀
#     for i in range(epoch):
#         gradient = numerical_gradient_training(f, x, data_training)
#
#         print(f'epoch={i + 1}, gradient={gradient}, x={x}')
#
#         x -= lr * gradient
#     return x