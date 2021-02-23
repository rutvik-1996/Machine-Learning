#Assignment 3: Gradient Boosting[Fashion MNIST]
#Submitted by: Rutvi Kantawala
#Instructor:Prof. Ajit Appari
#Class ALY6020 CRN 71778
#Aim:Develop a Adaptive and Gradient Boost based Decision Tree classifiers
#Output: Classify clothing items in the Fashion MNIST dataset

#Import dataset and read the files-already divided into train and test 
fashion_train=read.csv(file.choose(),header = FALSE)
dim(fashion_train)
fashion_test=read.csv(file.choose(),header =  FALSE)
dim(fashion_test)

#STEP 1:  Data Sampling
#use set seed to get same output every time program is run
set.seed(7777777)
#Use 30% data for analysis purpose in both train and test datasets
library(dplyr)
fmnist_train<- fashion_train %>% sample_frac(.3)
fmnist_test<- fashion_test %>% sample_frac(.3)
dim(fmnist_train)
dim(fmnist_test)


#STEP 2: Prediction Model Development & Validation
#Decision Tree Package: C50  {Boosting is done using AdaBoost equivalent technique}
#Convert integers to factors in column 1--v1 using as.factor
library(dataPreparation)
c <- as.factor(fmnist_train$V1)
c

install.packages("C50")  
library(C50)
#Model building
model1<-C5.0 (fmnist_train[,-1],c)
#We find tree size having 560 terminal nodes
model1
summary(model1)

#Predicting the accuracy--Validating model to check if future data is added how will it work
model1pred <- predict.C5.0(model1,fmnist_test[,-1],type = "class")
View(model1pred)

#Load required libraries
library(caret)
library(mlblench)
library(pROC)
library(class)
library(gmodels)
library(e1071)

#Checking accuracy through use of confusion matrix
confusionMatrix(table(model1pred,fmnist_test$V1))

#Experiment with different model parameters to produce the best version of decision tree
#Boosting using trials =5,10,25 such that new learners pick up the slack of older learners.  
#we can increase the model accuracy through Boosting process.
model5=C5.0 (fmnist_train[,-1],c,trials = 5)
model5
model5pred <- predict.C5.0(model5,fmnist_test[,-1],type = "class")
confusionMatrix(table(model5pred,fmnist_test$V1))

model10=C5.0 (fmnist_train[,-1],c,trials = 10)
model10
model10pred <- predict.C5.0(model10,fmnist_test[,-1],type = "class")
confusionMatrix(table(model10pred,fmnist_test$V1))

model25=C5.0 (fmnist_train[,-1],c,trials = 25)
model25
model25pred <- predict.C5.0(model25,fmnist_test[,-1],type = "class")
confusionMatrix(table(model25pred,fmnist_test$V1))

#XGBoost gradient boosting method
#Matrix creation for Training sample
labeltrain = fmnist_train$V1
summary.factor(labeltrain)   
intlabeltrain = as.integer(labeltrain)  
summary.factor(intlabeltrain)
install.packages("xgboost")
library(xgboost)
fashionxgb_train  <- xgb.DMatrix(data.matrix(fmnist_train[,-1]), label = intlabeltrain)

#Matrix creation for Test sample
labeltest = fmnist_test$V1
summary.factor(labeltest) 
intlabeltest = as.integer(labeltest) 
summary.factor(intlabeltest)
fashionxgb_test  <- xgb.DMatrix(data.matrix(fmnist_test[,-1]), label = intlabeltest)

#Train xgboost model using various nrounds to see if accuracy increases or not
parameters <- list(eta = 0.3, max_depth = 6, objective = "multi:softmax",  eval_metric = "merror")
fashion_xgboost <- xgboost(data=fashionxgb_train, nround=20,num_class=10, params=parameters)  
fashion_xgboost
fashion_xgboost1 <- xgboost(data=fashionxgb_train, nround=30,num_class=10, params=parameters)  
fashion_xgboost1

### Evaluation on test set
pred_fashionxgb <- predict(fashion_xgboost, fashionxgb_test, type="response")
View(pred_fashionxgb)
summary(pred_fashionxgb)
#Finding accuracy using confusion matrix
library(caret)
confusionMatrix(as.factor(intlabeltest), as.factor(pred_fashionxgb))

pred_fashionxgb1 <- predict(fashion_xgboost1, fashionxgb_test, type="response")
confusionMatrix(as.factor(intlabeltest), as.factor(pred_fashionxgb1))

