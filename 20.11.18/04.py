# 평균제곱오차

import sys
import numpy as np
import os
from pathlib import Path

try :
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'library'))
    from common import method_least_squares, mean_squares_Error
except :
    print('Library Module Can Not Found')

times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

a, b = method_least_squares(times, scores)

print(f'직선 y = {a}x + {b}')

print(f'오차(평균제곱오차) : {mean_squares_Error(np.array([a, b]), times, scores)}')