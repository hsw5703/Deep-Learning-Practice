# 최소제곱법(Method of Least Squares)
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

try :
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'library'))
    from common import method_least_squares
except :
    print('Library Module Can Not Found')

times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

a, b = method_least_squares(times, scores)

print(f'직선 y = {a}x + {b}')

scores_predict = [(a * t) + b for t in times]

fig, splt = plt.subplots()
splt.scatter(times, scores)
splt.plot(times, scores_predict)
plt.show()
