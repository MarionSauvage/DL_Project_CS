import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

def display_predictions(predictions):
    for i in range(len(predictions)):
        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        for j, ax in enumerate(axes.flat):
            # if j == 0:
            #     ax.imshow(Image.fromarray((predictions[i][j].transpose() * 255).astype(np.uint8)))
            ax.imshow(Image.fromarray((predictions[i][j + 1][0].transpose() * 255).astype(np.uint8)))
        plt.show()


def display_predictions2(predictions):
    fig=plt.figure(figsize=(50,50))
    outer=gridspec.GridSpec(5,4,wspace=0.2,hspace=0.2)
    for i in range(20):
        inner=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            t = ax.imshow(Image.fromarray((predictions[i][j + 1][0].transpose() * 255).astype(np.uint8)))
            #t.set_ha('center')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    plt.show() 
