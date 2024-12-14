import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt

df = pd.read_csv('Dataset/train_1.csv')  


sample_img = cv2.imread(os.path.join('Dataset/train_images/train_images', df['id_code'][0] + '.png'))
plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
plt.title(f"Label: {df['diagnosis'][0]}")
plt.show()
