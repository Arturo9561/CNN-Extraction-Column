# CNN-Extraction-Column

Machine_Learning_RF_Extraction_Classification-ErtuğrulFurkanDüzenli(Python File)

Libraries needed to be imported 
numpy 
matplotlib.pyplot
glob
cv2
os
seaborn 
pandas 
skimage.filters


-the images can be downloaded from-> https://tu-dortmund.sciebo.de/s/CmUAxIEv1vmriba
-Flooding and Regular_State have been used for the project
-The flooding and Regular_State data files have been splitted to a training folder (%70), 3616 images and testing folder (%30) 1542 images


in this line of the code your training data folder path should be included 
for directory_path in glob.glob("your/own/path_for_testing/*"):

in this line of the code your training data folder path should be included 
for directory_path in glob.glob("your/own/path_for_training/*"):

Transfer_Learning_VGG16_Extraction_Classification-ErtuğrulFurkanDüzenli(Python File)

Following libraries needed to be imported 
tensorflow
tensorflow/keras
tensorflow.keras/layers
tensorflow.keras.preprocessing.image/ImageDataGenerator
numpy
matplotlib.pyplot 
tensorflow.keras.models/Sequential
tensorflow.keras.layers/Dense, Conv2D, Flatten
tensorflow.keras/Model
tensorflow.keras.preprocessing.image/load_img, img_to_array

the images can be downloaded from-> https://tu-dortmund.sciebo.de/s/CmUAxIEv1vmriba
-Flooding and Regular_State have been used for the project
-The flooding and Regular_State data files have been splitted to a training folder (%70), 3616 images and testing folder (%30) 1542 images

in the following line of the codes your training/testing data folder path should be included 

training_batches tf.keras.preprocessing.image_dataset_from_directory(
    ("your/own/path_for_training/*"),...
testing_batches tf.keras.preprocessing.image_dataset_from_directory(
    ("your/own/path_for_testing/*), 

Also the path for the model evaluation steps need to be changed accordingly
Section 7 and 13

Other codes of the jupyter files of the project belongs to (Marvin Schwing)
additionally a package for keras.tuner should be installed

Native Parts of Marvin from the report:
-CNN Own Net
-Mathematical Approach, Bubble Detection with circleHough
Native Parts of Furkan from the report:
-CNN Pretrained Net
-Random Forest, gabor and sobel filters

The other parts of thereport were done with equal amount of work from both parties





