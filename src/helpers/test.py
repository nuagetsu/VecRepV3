from src.data_processing.ImageGenerators import get_island_image_set
import numpy as np
import matplotlib.pyplot as plt
arr = get_island_image_set("5island30max_ones", 1000)
arr = np.array (arr)
#np.save("C:/Users/Asus/PycharmProjects/VecRepV3/data/final_sample_small", arr)
plt.imshow(arr[0])
plt.show()
plt.imshow(arr[2])
plt.show()
plt.imshow(arr[5])
plt.show()
plt.imshow(arr[7])
plt.show()