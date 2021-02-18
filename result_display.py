import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_predictions(predictions):
    for i in range(len(predictions)):
        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        for j, ax in enumerate(axes.flat):
            # if j == 0:
            #     ax.imshow(Image.fromarray((predictions[i][j].transpose() * 255).astype(np.uint8)))
            ax.imshow(Image.fromarray((predictions[i][j + 1][0].transpose() * 255).astype(np.uint8)))
        plt.show()
