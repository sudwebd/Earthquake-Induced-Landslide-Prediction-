import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
x = np.arange(6)
x1 = [100472, 102090, 101895, 81358, 102651, 102912]
x2 = [2375, 1757, 1952, 23489, 1796, 1735]
plt.bar(x-0.2, x1,width=0.2,color='g',align='center')
plt.bar(x, x2,width=0.2,color='r',align='center')
plt.xticks(x, ['LR','KNN', 'ANN', 'NB', 'DT', 'RF'])
plt.show()
