import sys
import numpy as np
import os
from pathlib import Path

try :
    # sys.path.append('C:\Deep-Learning\Deep-Learning-Practice\library') # library가 담겨있는 위치를 가져온다.
    # from common import numerical_gradient  # 'common library에서 numerical_gradient 함수만 사용하겠다.'
    # import common as cm # 또는 'common library를 다 가져와서 사용하겠다.'

    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'library')) # 위치를 모르겠다고 하면 이렇게 사용할 수도 있다.
    from common import numerical_gradient

    # os.getcwd()는 현재 경로를 구하는 함수이다.

except :
    print('Library Module Can Not Found')

def f(x) :
    return np.sum(x**2, axis=0)

gra1 = numerical_gradient(f, np.array([3., 4.]))
gra2 = numerical_gradient(f, np.array([-1., -1.5]))
gra3 = numerical_gradient(f, np.array([-0.25, 0.25]))

print(gra1, gra2, gra3)