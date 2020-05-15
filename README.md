# COVID-19-CT-Classification
COVID 19 CT Image Classification
# The utility of this dataset has been confirmed by a senior radiologist in Tongji Hospital, Wuhan, China, who has performed   diagnosis and treatment of a large number of COVID-19 patients during the outbreak of this disease between January and April.

# Data Description
The COVID-CT-Dataset has 349 CT images containing clinical findings of COVID-19 from 216 patients. Source of this dataset is https://github.com/UCSD-AI4H/COVID-CT
Data Split as mentioned in the 'Datasplit' section of Github link
The Data Split is performed as mentioned in the above link. The split has been done in the below manner.

## Training dataset : {'Covid' : 191 , 'NonCovid' : 234 }
## Validation dataset : {'Covid : 60 , 'NonCovid : 58}
## Testing dataset : {'Covid' : 98 , 'NonCovid' : 105 }

# Architecture of the Model
CNN model has been used for image classification of the CT images of Covid and Non Covid Cases.

# Input ----> 2D Convolution ->Relu-2D MaxPooling-> Dropout----->Dense--Sigmoid------>Output
You can find the step by step guide for the model development in the Covid_Classifier_CNN.ipynb file.

# Let's begin with the CNN model development!

Suggestions for the improvement of this model are welcome.....
