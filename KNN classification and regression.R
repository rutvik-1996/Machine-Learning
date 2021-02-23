#Assignment 1: k-Nearest Neighbors [Fashion MNIST]
#Submitted by: Rutvi Kantawala
#Instructor:Prof. Ajit Appari
#Class ALY6020 CRN 71778
#Goal: Train and test image classification model of fashion dataset and tackle the classification problem using the K- nearest neighbors algorithm.
#Use K = 1, 11, and 21 to find accuracy and compare which is better value for k
#Compute and present the resulting confusion matrices. 
#Reference: https://www.edureka.co/blog/knn-algorithm-in-r/

#Install required packages
install.packages('caret')
install.packages('e1071')

#Load required libraries 
library(caret)
library(mlblench)
library(pROC)
library(class)
library(gmodels)
library(e1071)

#Import dataset and read the files-already divided into train and test 
fashion_train=read.csv(file.choose(),header = FALSE)
str(fashion_train)
head(fashion_train)

fashion_test=read.csv(file.choose(),header =  FALSE)
str(fashion_test)
head(fashion_test)

#Identify number of rows and columns in train and test data
#We find 60,000 rows in train data
dim(fashion_train)
#We find 10,000 rows in test data
dim(fashion_test)

#Check if there are any null values in dataset
sum(is.na(fashion_train))
sum(is.na(fashion_test))

set.seed(123)
#Creating train and test variables and scaling them
#Scaling helps to create first column as target variable as we don't have it in our dataset 
sample_train<-fashion_train[,-1]
sample_test<-fashion_test[,-1]

#Creating separate data-frame so final output can be compared with actual value
train1<-fashion_train[,1]
test1<-fashion_test[,1]


#KNN Model building
#Install class package
install.packages('class')
# Load class package
library(class)
prediction<-knn(train = sample_train,test = sample_test,cl=train1,k=1,prob = TRUE )
#Evaluating model performance
library(gmodels)
CrossTable(x = test1, y = prediction ,prop.chisq=FALSE) 

#Finding Accuracy of model-METHOD 1
#84.97% accurate
ACC.1<- 100 * sum(test1 == prediction)/NROW(test1)
print(ACC.1)

# Check prediction against actual value in tabular form for k=1
table(prediction ,test1)
#Checking accuracy through use of confusion matrix-METHOD 2
library(caret)
confusionMatrix(table(prediction ,test1))

#Predicting model by changing values of k to 11 
#84.76% accurate
prediction1<-knn(train = sample_train,test = sample_test,cl=train1,k=11,prob = TRUE )
CrossTable(x = test1, y = prediction1 ,prop.chisq=FALSE)
confusionMatrix(table(prediction1 ,test1))

#Predicting model by changing values of k to 21 
#84.02% accurate
prediction2<-knn(train = sample_train,test = sample_test,cl=train1,k=21,prob = TRUE )
CrossTable(x = test1, y = prediction2 ,prop.chisq=FALSE)
confusionMatrix(table(prediction2 ,test1))





