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
    return np.sum(x**2, axis=0)

# gradient_descent(f, np.array([-3., 4.]), lr=0.1) # 소수점을 붙여줘야 한다. 아니면 다른 변수형인 int형이 되어 Error.
# gradient_descent(f, np.array([-3., 4.]), lr=10)
gradient_descent(f, np.array([-3., 4.]), lr=0.001, epoch=10000)