

#!pip install exifread

"""##Import packages"""

import numpy as np
import os
import statistics
import pandas as pd
from PIL import ImageStat, Image
from scipy import stats
from scipy.stats import kurtosis,skew
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

mean_metadata = {}
mean_metadata_p = {}



def signaltonoise(img_array, axis=0, ddof=0):
    m = img_array.mean(axis)
    sd = img_array.std(axis=axis, ddof=ddof)
    return float(np.where(sd == 0, 0, m/sd))
  
def sparcity(img_array):
  sparsity = (np.product(img_array.shape)-np.count_nonzero(img_array))/np.product(img_array.shape)
  return sparsity


def get_class_pulmonary(file_name):
  firstTest = file_name.find('_')
  return int(file_name[file_name.find('_', firstTest + 1)+1: file_name.find('.')])


def get_metadata_pulmonary(name_dataset, dataset, subset, type_dataset, modality, task, color, number_classes):

    #get filenames
    file_names = os.listdir(os.getcwd())
    number_instances, number_instances_p = 0, 0
    width, height, skew_o, kurtosis_o, sparcity_o, snr = list(), list(), list(), list(), list(), list()
    width_p, height_p, skew_p, kurtosis_p, sparcity_p, snr_p = list(), list(), list(), list(), list(), list()

    for img_path in file_names:
        try:
            im = Image.open(img_path)
            img = np.asarray(im)
            img_class = get_class_pulmonary(img_path)
            if img_class == 0: #normal
                number_instances_p +=1
                width_p.append(im.width)
                height_p.append(im.height)
                skew_p.append(skew(im,axis=None))
                kurtosis_p.append(kurtosis(im,axis=None))
                sparcity_p.append(sparcity(img))
                snr_p.append(signaltonoise(img, axis=None))
                im.close
            elif img_class == 1: #abnormal
                number_instances +=1
                width.append(im.width)
                height.append(im.height)
                skew_o.append(skew(im,axis=None))
                kurtosis_o.append(kurtosis(im,axis=None))
                sparcity_o.append(sparcity(img))
                snr.append(signaltonoise(img, axis=None))
                im.close
        except IOError:
            pass
    
    #save normal classes
    global mean_metadata
    global mean_metadata_p

    mean_metadata_p = {'name': name_dataset,
                     'dataset':dataset,
                     'subset':subset, 
                     'type':type_dataset, 
                     'modality': modality,
                     'task':task, 
                     'class':1,
                     'width':statistics.mean(width_p), 
                     'height':statistics.mean(height_p),
                     'skew':statistics.mean(skew_p),
                     'kurtosis':statistics.mean(kurtosis_p),
                     'sparcity':statistics.mean(sparcity_p), 
                     'signal_noise':statistics.mean(snr_p),
                     'color':color,
                     'number_instances':number_instances_p,
                     'number_classes':number_classes
                     }

    #save abnormal classes
    mean_metadata = {'name': name_dataset,
                     'dataset':dataset,
                     'subset':subset, 
                     'type':type_dataset, 
                     'modality': modality,
                     'task':task, 
                     'class':2,
                     'width':statistics.mean(width), 
                     'height':statistics.mean(height),
                     'skew':statistics.mean(skew_o),
                     'kurtosis':statistics.mean(kurtosis_o),
                     'sparcity':statistics.mean(sparcity_o), 
                     'signal_noise':statistics.mean(snr),
                     'color':color,
                     'number_instances':number_instances,
                     'number_classes':number_classes
                     }



def get_metadata(name_dataset, dataset, subset, type_dataset, modality, task, class_dataset, color, number_classes):

    #get filenames
    file_names = os.listdir(os.getcwd())
    number_instances = len(file_names)
    width, height, skew_o, kurtosis_o, sparcity_o, snr = list(), list(), list(), list(), list(), list()

    for img_path in file_names:
        try:
            im = Image.open(img_path)
            img = np.asarray(im)
            width.append(im.width)
            height.append(im.height)
            skew_o.append(skew(im,axis=None))
            kurtosis_o.append(kurtosis(im,axis=None))
            sparcity_o.append(sparcity(img))
            snr.append(signaltonoise(img, axis=None))
            im.close
        except IOError:
            pass

    save_metadata(name_dataset, dataset, subset, type_dataset, modality, task, class_dataset, width, height, 
                  skew_o, kurtosis_o, sparcity_o, snr, color, number_instances, number_classes)


def save_metadata(name_dataset, dataset, subset, type_dataset, modality, task, class_dataset, width, height, 
                  skew_o, kurtosis_o, sparcity_o, signal_noise_o, color, number_instances, number_classes):

    global mean_metadata
    mean_metadata = {'name': name_dataset,
                     'dataset':dataset,
                     'subset':subset, 
                     'type':type_dataset, 
                     'modality': modality,
                     'task':task, 
                     'class':class_dataset,
                     'width':statistics.mean(width), 
                     'height':statistics.mean(height),
                     'skew':statistics.mean(skew_o),
                     'kurtosis':statistics.mean(kurtosis_o),
                     'sparcity':statistics.mean(sparcity_o), 
                     'signal_noise':statistics.mean(signal_noise_o),
                     'color':color,
                     'number_instances':number_instances,
                     'number_classes':number_classes
                     }



columnsTitles = ['name', 'dataset', 'subset', 'type', 'modality', 'task', 'class', 'width', 
'height', 'skew', 'kurtosis', 'sparcity', 'signal_noise', 'color', 'number_instances', 'number_classes']


#chest-xray test-pneumonia#

#testing data
os.chdir('/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/chest_xray/test/NORMAL')
get_metadata('chest-xray-pneumonia', 1, 1, 1, 1, 1, 1, 1, 2)
chest_xray_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
chest_xray_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/results/test-normal.csv')

os.chdir('/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/chest_xray/test/PNEUMONIA')
get_metadata('chest-xray-pneumonia', 1, 1, 1, 1, 1, 2, 1, 2)
chest_xray_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
chest_xray_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/results/test-pneumonia.csv')

#training data
os.chdir('/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/chest_xray/train/NORMAL')
get_metadata('chest-xray-pneumonia', 1, 2, 1, 1, 1, 1, 1, 2)
chest_xray_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
chest_xray_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/results/train-normal.csv')

os.chdir('/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')
get_metadata('chest-xray-pneumonia', 1, 2, 1, 1, 1, 2, 1, 2)
chest_xray_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
chest_xray_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/results/train-pneumonia.csv')

#validation data
os.chdir('/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/chest_xray/val/NORMAL')
get_metadata('chest-xray-pneumonia', 1, 3, 1, 1, 1, 1, 1, 2)
chest_xray_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
chest_xray_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/results/val-normal.csv')

os.chdir('/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/chest_xray/val/PNEUMONIA')
get_metadata('chest-xray-pneumonia', 1, 3, 1, 1, 1, 2, 1, 2)
chest_xray_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
chest_xray_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/chest-xray-pneumonia/results/val-pneumonia.csv')




#pulmonary-chest-xray-abnormalities#
#Montgomery
os.chdir('/home/garciaas/meta-learning/datasets/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png')
get_metadata_pulmonary('pulmonary-chest-xray-abnormalities', 2, 4, 1, 1, 1, 1, 2)
pulmonary_metadata_p = pd.DataFrame([mean_metadata_p], columns=columnsTitles)
pulmonary_metadata_p.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/pulmonary-chest-xray-abnormalities/results/montgomery-normal.csv')
pulmonary_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
pulmonary_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/pulmonary-chest-xray-abnormalities/results/montgomery-abnormal.csv')

#ChinaSet
os.chdir('/home/garciaas/meta-learning/datasets/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png')
get_metadata_pulmonary('pulmonary-chest-xray-abnormalities', 2, 5, 1, 1, 1, 1, 2)
pulmonary_metadata_p = pd.DataFrame([mean_metadata_p], columns=columnsTitles)
pulmonary_metadata_p.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/pulmonary-chest-xray-abnormalities/results/chinaset-normal.csv')
pulmonary_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
pulmonary_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/pulmonary-chest-xray-abnormalities/results/chinaset-abnormal.csv')


#blood-cells/

#TEST/EOSINOPHIL
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TEST/EOSINOPHIL')
get_metadata('blood-cells', 3, 1, 2, 2, 1, 3, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test-eosinophil.csv')

#TEST/LYMPHOCYTE
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TEST/LYMPHOCYTE')
get_metadata('blood-cells', 3, 1, 2, 2, 1, 4, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test-lymphocyte.csv')

#TEST/LYMPHOCYTE
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TEST/MONOCYTE')
get_metadata('blood-cells', 3, 1, 2, 2, 1, 5, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test-monocyte.csv')

#TEST/NEUTROPHIL
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TEST/MONOCYTE')
get_metadata('blood-cells', 3, 1, 2, 2, 1, 6, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test-neutrophil.csv')


#TEST_SIMPLE/EOSINOPHIL
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TEST_SIMPLE/EOSINOPHIL')
get_metadata('blood-cells', 3, 3, 2, 2, 1, 3, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test_simple-eosinophil.csv')

#TEST_SIMPLE/LYMPHOCYTE
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TEST_SIMPLE/LYMPHOCYTE')
get_metadata('blood-cells', 3, 3, 2, 2, 1, 4, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test_simple-lymphocyte.csv')

#TEST_SIMPLE/LYMPHOCYTE
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TRAIN/MONOCYTE')
get_metadata('blood-cells', 3, 3, 2, 2, 1, 5, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test_simple-monocyte.csv')

#TEST_SIMPLE/NEUTROPHIL
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TEST_SIMPLE/MONOCYTE')
get_metadata('blood-cells', 3, 2, 3, 2, 1, 6, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/test_simple-neutrophil.csv')


#TRAIN/EOSINOPHIL
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TRAIN/EOSINOPHIL')
get_metadata('blood-cells', 3, 2, 2, 2, 1, 3, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/train-eosinophil.csv')

#TRAIN/LYMPHOCYTE
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TRAIN/LYMPHOCYTE')
get_metadata('blood-cells', 3, 2, 2, 2, 1, 4, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/train-lymphocyte.csv')

#TRAIN/LYMPHOCYTE
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TRAIN/MONOCYTE')
get_metadata('blood-cells', 3, 2, 2, 2, 1, 5, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/train-monocyte.csv')

#TRAIN/NEUTROPHIL
os.chdir('/home/garciaas/meta-learning/datasets/blood-cells/dataset2-master/dataset2-master/images/TRAIN/MONOCYTE')
get_metadata('blood-cells', 3, 2, 2, 2, 1, 6, 2, 4)
blood_metadata = pd.DataFrame([mean_metadata], columns=columnsTitles)
blood_metadata.to_csv(index=False, path_or_buf='/home/garciaas/meta-learning/datasets/blood-cells/results/train-neutrophil.csv')








'''{
    name: 
        1='chest-xray-pneumonia'
        2='pulmonary-chest-xray-abnormalities'
        3='blood-cells'
    subset:
        1='test'
        2='train'
        3='val' --test-simple/blood-cells 
        4='Montgomery/pulmonary'
        5='China/pulmonary'  
    type:
        1='chest, lung'
        2='histopatology'
    modality:
        1='xray'
        2='microscope'
    task:
        1='classification'
    class:
        1='negative, normal'
        2='positive, disease'
        3='EOSINOPHIL/blodd-cells'
        4='LYMPHOCYTE/boold-cells'
        5='MONOCYTE/blood-cells'
        6='NEUTROPHIL/blood-cells'
    color:
        1='mode L grayscale'
        2='color RGB'
}'''