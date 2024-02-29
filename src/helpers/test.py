import numpy as np
from sklearn.preprocessing import normalize
import src.data_processing.VecRep as utils

arr = np.array([[0,2,3],[0,5,6],[7,8,9]])
print(normalize(arr, norm='l2', axis=0))
