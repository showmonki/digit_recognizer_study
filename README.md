# Digit Recognizer Study Notes

- 此为个人学习记录使用
- keras version



## Kaggle data

Data Overview:

* Image
* 28x28 (784 pixels, 0-255) 
* label 0-9

Train:

* 785 columns, 'labels'(first col) with pixels

- decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive.
- Distribution quite even, without missing info. Each digit around 4k.



submisssion:

* ImageID, Label

> kaggle: submission scored 0.98832

# Code

### preprocess

* to_categorical

### mdoel 1: base model

fc with 3 layers

### model 2: cnn 

# Version

python版本为3.7



## TODO

- [ ] data explore convert to ipynb
- [ ] model flowchart drawid
- [ ] print model summary + model performance, records # as config
- [ ] save model
- [ ] system web like with prob 
- [x] generate requirements.txt 

