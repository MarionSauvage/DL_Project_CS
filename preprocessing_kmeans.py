import numpy as np
import pandas as pd
import os
from PIL import Image

def load_dataset(data_path):
    # Go through all files and create dictionary containing the data
    images_dict = {}
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if 'tif' in name or 'png' in name:
                file_path = os.path.join(path, name)

                # Load image to store it as a numpy array in the dict
                img = Image.open(file_path)

                if 'mask' in name:
                    img_id = name[:-9]               

                    if img_id in images_dict:
                        images_dict[img_id]['mask_path'] = file_path
                    else:
                        images_dict[img_id] = {'mask_path': file_path}

                    images_dict[img_id]['mask_data'] = np.array(img)
                else:
                    img_id = name[:-4]

                    # Only keep the FLAIR part of the tif image
                    img_array = np.array(img)
                    img_flair_array = img_array[:, :, 1]

                    if img_id in images_dict:
                        images_dict[img_id]['image_path'] = file_path
                    else:
                        images_dict[img_id] = {'image_path': file_path}
                    
                    images_dict[img_id]['image_data'] = np.array(img_flair_array)

    # Create pandas dataframe from images_dict
    dataset = pd.DataFrame.from_dict(images_dict, orient='index').reset_index()
    dataset = dataset.rename(columns={'index': 'patient_id'})
    print(dataset.head())

    # Delete images_dict from memory
    images_dict.clear()

    # Add target values
    dataset['tumor'] = dataset['mask_data'].apply(lambda x: 1 if np.max(x) > 0 else 0)

    return dataset
