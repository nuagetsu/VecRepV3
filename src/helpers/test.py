import numpy as np
from sklearn.preprocessing import normalize
import src.data_processing.VecRep as utils
import pprint

arr = np.array([[1.17437573, 0.86387531, 0.86387531, 0.17437573],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.17437573, 0.86387531, 0.86387531, 1.17437573]])
arr = normalize(arr, norm='l2', axis=0)
pprint.pprint(np.matmul(arr.T, arr))

