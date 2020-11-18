# 경사하강법

import sys
import numpy as np
import os
from pathlib import Path

try :
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'library'))
    from common import method_least_squares, mean_squares_Error, gradient_descent_linear_regression
except :
    print('Library Module Can Not Found')

times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

result = gradient_descent_linear_regression(mean_squares_Error, np.array([0., 0.]), epoch = 5000, data_training = (times, scores))
print(result)