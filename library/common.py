import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        h1 = f(x)

        x[i] = tmp - h
        h2 = f(x)

        gradient[i] = (h1 - h2) / (2 * h)
        x[i] = tmp

        return gradient


def gradient_descent(f, x, lr=0.01, epoch=100):  # 경사하강법
    for i in range(epoch):
        gradient = numerical_gradient(f, x)

        print(f'epoch={i + 1}, gradient={gradient}, x={x}')

        x -= lr * gradient
    return x


def method_least_squares(x, y):  # 최소제곱법
    mx = sum(x) / len(x)
    my = sum(y) / len(y)

    mls_a = sum([(i - mx) * (j - my) for i, j in zip(x, y)]) / sum([(i - mx) ** 2 for i in x])
    mls_b = my - (mx * mls_a)

    return mls_a, mls_b


def mean_squares_Error(x, data_x, data_y):  # 평균제곱오차(MSE, Mean Squares Error)
    s = 0
    for i in range(len(data_x)):
        data_y_hat = x[0] * data_x[i] + x[1]
        s += (data_y_hat - data_y[i]) ** 2
    e = s / len(data_x)

    data_y_hat = [x[0] * dx + x[1] for dx in data_x]
    e = np.mean([(dyh - dy) ** 2 for dyh, dy in zip(data_y_hat, data_y)])
    return e
