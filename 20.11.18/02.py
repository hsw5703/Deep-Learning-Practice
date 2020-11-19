# 경사하강법

import sys
import numpy as np
import os
from pathlib import Path

try :
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'library'))
    from common import gradient_descent
except :
    print('Library Module Can Not Found')

def f(x) :
    print(np.sum(x**2, axis=0))
    return np.sum(x**2, axis=0)
    # return x**2

# gradient_descent(f, np.array([-3., 4.]), lr=0.1) # 소수점을 붙여줘야 한다. 아니면 다른 변수형인 int형이 되어 Error.
# gradient_descent(f, np.array([-3., 4.]), lr=10)
gradient_descent(f, np.array([-3., -2]), lr=0.001, epoch=10)

# epoch나 lr의 값을 지정해주지 않으면 선언된 함수의 수로 진행된다.
# lr(learning rate : 학습률) : 갱신하는 양을 결정하는 변수. 얼마만큼 데이터를 이동시켜 최소 기울기를 찾을지 결정.
# epoch : 경사하강법의 반복 횟수.
