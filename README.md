## A Convolutional Neural Network for Predicting Ideology from Tweets

It adopts [Denny's][2] implementation of Kim's [Convolutional Neural Network for Sentence Classification][1]paper in tensor flow. 

### Requirement
* Python 2.7
* Numpy
* Tensor Flow ( CPU only, Python 2.7) ([installation Link][3])

### Files
1. data_helpers_pre.py 
   : Clean the data and make it into right format to be handled by CNN classifier

2. predict.py
   : Make predictions for input data

3. runs/
   : Folder contains pretrained CNN classifier and word2vec trained within CNN framework

4. text_cnn.py
   : Model Formulation

### Inputs
Sequence of tweets in string format

### Outputs
* 1 for Repbulican
* 0 for Democratic

### How to use model to predict
Modify data source in predict.py
then run python predict.py

### Note and TODO
1. Install the required libraries
2. Modified code in data_helpers_pre.py to read tweet from DynamoDB(if applicable), the output should be strings or sequence of strings
3. Modified code in the Data Loading section of predict.py to specify the input files 
4. Modified code at the end of predict.py to write prediction into DynamoDB
5. If batch process, please activate the bath processing in the predict.py
 
[1]: https://arxiv.org/abs/1408.5882   
[2]: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/README.md
[3]: https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html

