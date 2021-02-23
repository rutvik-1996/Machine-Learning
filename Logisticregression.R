#Assignment 2: Logistic regression [MNIST]
#Submitted by: Rutvi Kantawala
#Instructor:Prof. Ajit Appari
#Class ALY6020 CRN 71778
#####Import Dataset#####
mnist_train=read.csv(file.choose(),header =  FALSE)
mnist_test=read.csv(file.choose(),header =  FALSE)
library(dplyr)

#STEP 1
#use set seed to get same output every time program is run
set.seed(123)
#Use 30% data for analysis purpose in both train and test datasets
mnist_trainsubset<- mnist_train %>% sample_frac(.3)
mnist_testsubset<- mnist_test %>% sample_frac(.3)
dim(mnist_trainsubset)
dim(mnist_testsubset)

#STEP 2
#Relabel the first column showing handwritten digits in both train and test as TargetLabel
colnames(mnist_trainsubset)[1]<- "Targetlabel"
colnames(mnist_testsubset)[1]<- "Targetlabel"

# Create dummy labels for each digit using Data Preparation library
install.packages("dataPreparation")
library(dataPreparation)

#Changing label for train data
#converts the numeric column into factor
mnist_trainsubset<-setColAsFactor(mnist_trainsubset, cols="Targetlabel") 
#Encodes the values of "Targetlabel"
l1<- build_encoding(mnist_trainsubset, cols="Targetlabel")
View(l1)
#creates dummy columns for each digits
mnist_trainsubset<- one_hot_encoder(mnist_trainsubset, encoding=l1)
View(mnist_trainsubset)  

#Changing label for test data
mnist_testsubset<-setColAsFactor(mnist_testsubset, cols="Targetlabel")  
l2<- build_encoding(mnist_testsubset, cols="Targetlabel")
View(l2)
mnist_testsubset<- one_hot_encoder(mnist_testsubset, encoding=l2)
View(mnist_testsubset)  

#run 10 separate logistic regression using the glm command on training subset
library(nnet)

#Find columns with constant values
whichAreConstant(mnist_trainsubset)
install.packages("janitor")
library(janitor)
remove_constant(mnist_trainsubset, na.rm = FALSE, quiet = TRUE)

whichAreConstant(mnist_testsubset)
remove_constant(mnist_testsubset, na.rm = FALSE, quiet = TRUE)

#Model building for digit 0
model0 <- glm(Targetlabel.0 ~ .-Targetlabel.1-Targetlabel.2-Targetlabel.3-Targetlabel.4-Targetlabel.5-Targetlabel.6-Targetlabel.7-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model0)
#Testing the model for validation purpose
predictedmodel0 <- predict(model0,data=mnist_testsubset,type="response")
library(gmodels)
#Finding probabilities that are uncorelated
#Predict command will give you probability of each observation being digit "k". 
softmax<- exp(predictedmodel0)/sum(exp(predictedmodel0))
softmax

#Model building for digit 1
model1 <- glm(Targetlabel.1 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.3-Targetlabel.4-Targetlabel.5-Targetlabel.6-Targetlabel.7-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model1)
#Testing the model for validation purpose
predictedmodel1 <- predict(model1,data=mnist_testsubset,type="response")

#Model building for digit 2
model2 <- glm(Targetlabel.2 ~ .-Targetlabel.0-Targetlabel.1-Targetlabel.3-Targetlabel.4-Targetlabel.5-Targetlabel.6-Targetlabel.7-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model2)
#Testing the model for validation purpose
predictedmodel2 <- predict(model2,data=mnist_testsubset,type="response")

#Model building for digit 3
model3 <- glm(Targetlabel.3 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.1-Targetlabel.4-Targetlabel.5-Targetlabel.6-Targetlabel.7-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model3)
#Testing the model for validation purpose
predictedmodel3 <- predict(model3,data=mnist_testsubset,type="response")

#Model building for digit 4
model4 <- glm(Targetlabel.4 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.3-Targetlabel.1-Targetlabel.5-Targetlabel.6-Targetlabel.7-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model4)
#Testing the model for validation purpose
predictedmodel4 <- predict(model4,data=mnist_testsubset,type="response")

#Model building for digit 5
model5 <- glm(Targetlabel.5 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.3-Targetlabel.4-Targetlabel.1-Targetlabel.6-Targetlabel.7-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model5)
#Testing the model for validation purpose
predictedmodel5 <- predict(model5,data=mnist_testsubset,type="response")

#Model building for digit 6
model6 <- glm(Targetlabel.6 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.3-Targetlabel.4-Targetlabel.5-Targetlabel.1-Targetlabel.7-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model6)
#Testing the model for validation purpose
predictedmodel6 <- predict(model6,data=mnist_testsubset,type="response")

#Model building for digit 7
model7 <- glm(Targetlabel.7 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.3-Targetlabel.4-Targetlabel.5-Targetlabel.6-Targetlabel.1-Targetlabel.8-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model7)
#Testing the model for validation purpose
predictedmodel7 <- predict(model7,data=mnist_testsubset,type="response")

#Model building for digit 8
model8 <- glm(Targetlabel.8 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.3-Targetlabel.4-Targetlabel.5-Targetlabel.6-Targetlabel.7-Targetlabel.1-Targetlabel.9,family = binomial(link=logit),data = mnist_trainsubset)
summary(model8)
#Testing the model for validation purpose
predictedmodel8 <- predict(model8,data=mnist_testsubset,type="response")

#Model building for digit 9
model9 <- glm(Targetlabel.9 ~ .-Targetlabel.0-Targetlabel.2-Targetlabel.3-Targetlabel.4-Targetlabel.5-Targetlabel.6-Targetlabel.7-Targetlabel.8-Targetlabel.1,family = binomial(link=logit),data = mnist_trainsubset)
summary(model9)
#Testing the model for validation purpose
predictedmodel9 <- predict(model9,data=mnist_testsubset,type="response")









