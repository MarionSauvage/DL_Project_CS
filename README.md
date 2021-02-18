# DL_Project_CS : Brain MRI classification and segmentation

This project is part of a Deep Learning course in AI major at CentraleSupélec.

## Clone repository

```sh
$ git clone https://github.com/MarionSauvage/DL_Project_CS.git
```

## Dataset download
Data hosted on kaggle : https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.
Tumor genomic clusters and patient data is provided in data.csv file.
There is no API. It has to be uplaod manually.

![image](images/dataset.png)



## Repository structure

``` bash 
├── README.md
├── classification
│   ├── classification.py
│   ├── classification_optimization.py
│   ├── model_classification.py
│   └── preprocessing_classification.py
│── images
│   ├── dataset.png
│   └── unet_archi.png   
├── segmentation
│   ├── preprocessing_segmentation.py
│   ├── model_segmentation.py
│   ├── result_display.py
│   └── segmentation.py
├── data_visualization.ipynb
├── main_classification.py
├── main_segmentation.py
├── preprocessing.py
└── .gitignore
```

* In the root directory, we have :
   - `preprocessing.py` which is used to process the dataset before applying classification or segmentation codes. 
    - There is as well a `data_vizualisation.ipynb`, a jupyter notebook allowing to get a better understanding of the dataset.
    - There 2 main files `main_classification.py`and `main_segmentation.py` which respectively allow to perform classification and semgentation on the dataset. 

* In the **images** directory, one find the images present in the README.

* In **classification** directory :
    - ``classification.py``
    - ``classification_optimization.py``
    - ``model_classification.py``
    - ``preprocessing_classification.py``

* In **segmentation** directory 

## Requirements 



## Models

Segmentation with UNET architecture

![image](images/unet_archi.png)

Source : https://becominghuman.ai/implementing-unet-in-pytorch-8c7e05a121b4
