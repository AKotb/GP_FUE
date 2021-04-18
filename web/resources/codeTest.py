# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 06:21:35 2021

@author: 20115
"""
import sys, getopt
import pdb
import logging
import numpy as np
import os
from time import gmtime, strftime
import pickle

from os import listdir
from os.path import isfile, join

import multiprocessing 
from random import randint

from docopt import docopt
from osgeo import gdal
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from astropy import online_docs_root

scenes=['delta','fayoum']
years=['2010','2011']
algorithm=['rf','svm','mlp','nn','gp','dt','rf2','mlp2','ap','qd']
dataset=['Training','Testing']

#List takes 4 parameters.
myopts, args = getopt.getopt(sys.argv[1:], "s:y:g:n:")
for o, t in myopts:
    if o == '-s':
        s = t
    elif o == '-y':
        y = t
    elif o == '-g':
        a = t
    elif o == '-n':
        u = t
    else:
        # If there is a wrong in any of the parameters, the system will output this message.
        print("Usage: %s -s scene -y year -g algorithm -n username " % sys.argv[0])

# Display input and output dir name passed as the args.
# If the length of the parameters is 4 then do this function.
if len(myopts) == 4:
    s = int(s)
    y = int(y)
    a = int(a)
    username = u
    logger = logging.getLogger(__name__)
else:
    # If the length of the parameters is not 4 then print out this message.
    print("Usage: %s -s scene -y year -g algorithm -n username " % sys.argv[0])

'''
s=int(input("plz enter scene"))
y=int(input('plz enter year'))
a=int(input('plz enter algorithm'))
username=input('plz enter your name ')
logger = logging.getLogger(__name__)
'''

#def report_and_exit(txt, *args, **kwargs):
    #logger.error(txt, *args, **kwargs)
    #exit(1)

CLASSIFIERS = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'rf': RandomForestClassifier(n_jobs=4, n_estimators=10, class_weight='balanced'),
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        'svm': SVC(class_weight='balanced' , C = 1000.0 , gamma= 0.00000001),
        # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        'mlp':  MLPClassifier(activation='relu',
                                  alpha=1e-4,
                                  batch_size='auto', 
                                  beta_1=0.9,
                                  beta_2=0.999, 
                                  early_stopping=False, 
                                  epsilon=1e-08,
                                  hidden_layer_sizes=(9,), 
                                  learning_rate='constant',
                                  learning_rate_init=.1, 
                                  max_iter=200, 
                                  momentum=0.9,
                                  nesterovs_momentum=True, 
                                  power_t=0.5, 
                                  random_state=1, 
                                  shuffle=True,
                                  solver='adam', 
                                  tol=0.0001, 
                                  validation_fraction=0.1, 
                                  verbose=200,
                                  warm_start=True),
        'nn': KNeighborsClassifier(5),
        'gp': GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        'dt': DecisionTreeClassifier(max_depth=5),
        'RF2': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        'mlp2': MLPClassifier(alpha=1),
        'ap': AdaBoostClassifier(),
        'qd': QuadraticDiscriminantAnalysis()
    }

def algo_main():
    if a==0:
        classifier=RandomForestClassifier(n_jobs=4, n_estimators=10, class_weight='balanced')
    elif a==1:
        classifier =SVC(class_weight='balanced' , C = 1000.0 , gamma= 0.00000001)
    elif a==2:
        classifier=MLPClassifier(activation='relu',alpha=1e-4,
                                 batch_size='auto',beta_1=0.9,
                                 beta_2=0.999,early_stopping=False, 
                                 epsilon=1e-08, hidden_layer_sizes=(9,), 
                                 learning_rate='constant',learning_rate_init=.1, 
                                 max_iter=200,momentum=0.9,
                                 nesterovs_momentum=True,power_t=0.5, 
                                 random_state=1,shuffle=True,solver='adam',tol=0.0001, 
                                 validation_fraction=0.1,verbose=200,warm_start=True)
    elif a==3:
       classifier=KNeighborsClassifier(5)
    elif a==4:
       classifier= GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    elif a==5:   
       classifier= DecisionTreeClassifier(max_depth=5)
    elif a==6:  
       classifier=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    elif a==7:   
       classifier= MLPClassifier(alpha=1)
    elif a==8:
       classifier= AdaBoostClassifier()
    elif a==9:
       classifier=QuadraticDiscriminantAnalysis()
       
    #logger.debug("Train the classifier: %s", str(classifier))
    classifier.fit(training_samples, training_labels)
    predicted_labels = classifier.predict(testing_samples)

    res = "\n##############################################\n"
    res = res + output_rname
    res = res + "\n"  + str( CLASSIFIERS[method])
    
    target_names = ['Class %s' % s for s in classes]
    
    res = res + "\n" +  str("target_names:\n%s" %target_names)
    aa =    metrics.confusion_matrix(verification_labels, predicted_labels)
    
    res = res + "\n" +  str("Confussion matrix:\n%s" %
    metrics.confusion_matrix(verification_labels, predicted_labels))
           
    res = res + "\n" + str("Classification report:\n%s" %
    metrics.classification_report(verification_labels, predicted_labels,target_names=target_names))  
             
    q = metrics.accuracy_score(verification_labels, predicted_labels)
    res = res+ "\n" + str("Classification accuracy: %f" %
                    metrics.accuracy_score(verification_labels, predicted_labels))
                
    output_name = str(int(q*1000) ) + "_"+ output_rname 
    f = open(output_path + username + ".txt" , 'w')
    f.write(res)
   
    with open(output_path + username +".pickle", 'wb') as p :  # Python 3: open(..., 'wb')
        pickle.dump([classifier,classes],p)
    return q       

if __name__ == "__main__":
    #logging.info("Begin")   

    method = algorithm[a] # a= index from website    

    #datapth              =  "D:/GP/ml/scene/"+scenes[s]+'/'+years[y] #s=index of scene and y=year from website 
    train_data_path      =  "D:/GPWebsite/ml/pickle/"+scenes[s]+dataset[0]      
    validation_data_path =  "D:/GPWebsite/ml/pickle/"+scenes[s]+dataset[1]
    
    output_path  ="D:/GPWebsite/ml/result/"+username+'/'+algorithm[a]+'/'  
       
    if not os.path.exists(output_path):
       os.makedirs(output_path)
    onlyfiles = [f for f in listdir(output_path) if isfile(join(output_path, f))]
    output_rname         =    str(len(onlyfiles)+1 )  
    
    output_rname   = algorithm[a]

    with open(train_data_path+'.pickle',"rb") as t:  # Python 3: open(..., 'wb')
        training_samples, training_labels,classes = pickle.load(t)
                  
    with open(validation_data_path+'.pickle',"rb") as v:  # Python 3: open(..., 'wb')
        testing_samples, verification_labels = pickle.load(v)

algo_main()

#logger = logging.getLogger(__name__)

COLORS = [
    
    "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58","#C895C5"
]

def write_geotiff(fname, data, geo_transform, projection, data_type=gdal.GDT_Byte):

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)

    ct = gdal.ColorTable()
    for pixel_value in range(len(classes)+1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return

#def report_and_exit(txt, *args, **kwargs):
   # logger.error(txt, *args, **kwargs)
    #exit(1)
    
def predict_patch(single_sub_flat,classifier,i):
    sub_result = classifier.predict(single_sub_flat)
    return sub_result

def Parallel_manager(flat_pixels,classifier):
    sub_flat_pixels = np.array_split(flat_pixels,50)
    result = [] 
    i = 1
    for single_sub_flat in sub_flat_pixels:
        sub_result=predict_patch(single_sub_flat,classifier,i)
        result = np.append(result, sub_result)              
        i = i+1    
    return result    

if __name__ == "__main__":
    #logging.info("Begin")


    raster_data_path     = "D:/GPWebsite/ml/scene/"+scenes[s]+'/'+years[y]+'.tif' #s=index of scene and y=year from website 
    output_path          = "D:/GPWebsite/ml/result/"+username+ '/'+algorithm[a]
    classifier_file      =username+'.pickle'
    classifier_path_file =  output_path +'/'+ classifier_file
    
    raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
  
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []

    for b in range(1, raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())    

    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape
    n_samples = rows*cols  
    logger.debug("Classifing...")
    flat_pixels = bands_data.reshape((n_samples, n_bands))
    
    with open(classifier_path_file,'rb') as p:  # Python 3: open(..., 'wb')
           classifier,classes = pickle.load(p)
                      
    result = Parallel_manager(flat_pixels,classifier)

    classification = result.reshape((rows, cols))
    write_geotiff(classifier_path_file +".png" , classification, geo_transform, proj)
   