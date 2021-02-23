#Assignment 4: Neural network [Fashion MNIST]
#Submitted by: Rutvi Kantawala
#Instructor:Prof. Ajit Appari
#Class ALY6020 CRN 71778
#Aim:Train a feed forward neural network to classify each image into one of ten digits representing the clothing/accessory type.

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
fmnist_train<- fashion_train %>% sample_frac(.01)
fmnist_test<- fashion_test %>% sample_frac(.01)
dim(fmnist_train)
dim(fmnist_test)


#create 7 indicators to represent LED style pattern 
#Adding 7 new columns a-g with respective values in train data
df_train<-fmnist_train %>% 
  mutate(a=case_when(V1==1|V1==4 ~ "0",V1!=1|V1!=4 ~ "1"),
  b=case_when(V1==5|V1==6 ~ "0",V1!=5|V1!=6 ~ "1"),
  c=case_when(V1==2 ~ "0",V1!=2 ~ "1"),
  d=case_when(V1==1|V1==4|V1==7|V1==9 ~ "0",V1!=1|V1!=4|V1!=7|V1!=9 ~ "1"),
  e=case_when(V1==1|V1==3|V1==4|V1==5|V1==7|V1==9 ~ "0",V1!=1|V1!=3|V1!=4|V1!=5|V1!=7| V1!=9 ~ "1"),
  f=case_when(V1==1|V1==2|V1==3|V1==7 ~ "0",V1!=1|V1!=2|V1!=3|V1!=7 ~ "1"),
  g=case_when(V1==0|V1==1|V1==7 ~ "0",V1!=0|V1!=1|V1!=7 ~ "1"))

#Adding 7 new columns a-g with respective values in test data 
df_test<-fmnist_test %>% 
  mutate(a=case_when(V1==1|V1==4 ~ "0",V1!=1|V1!=4 ~ "1"),
         b=case_when(V1==5|V1==6 ~ "0",V1!=5|V1!=6 ~ "1"),
         c=case_when(V1==2 ~ "0",V1!=2 ~ "1"),
         d=case_when(V1==1|V1==4|V1==7|V1==9 ~ "0",V1!=1|V1!=4|V1!=7|V1!=9 ~ "1"),
         e=case_when(V1==1|V1==3|V1==4|V1==5|V1==7|V1==9 ~ "0",V1!=1|V1!=3|V1!=4|V1!=5|V1!=7| V1!=9 ~ "1"),
         f=case_when(V1==1|V1==2|V1==3|V1==7 ~ "0",V1!=1|V1!=2|V1!=3|V1!=7 ~ "1"),
         g=case_when(V1==0|V1==1|V1==7 ~ "0",V1!=0|V1!=1|V1!=7 ~ "1"))


#Model Building
install.packages("neuralnet")  
library(neuralnet)
datakeras <- paste(c(colnames(df_train[,-c(1,786:792)])),collapse="+")
datakeras <- paste(c("a+b+c+d+e+f+g ~",datakeras),collapse="")

model1 <- neuralnet(datakeras, 
                    df_train, hidden=1, rep = 2, err.fct = "ce", 
                    linear.output = FALSE, lifesign = "minimal", stepmax = 1000,
                    threshold = 0.001)

#Model plotting
plot(model1, rep="best")

df_test1<-df_test[, -c(786:792)]
df_test2<- subset(df_test1, select = -V1 )

#Model prediction
prediction <- compute(model1, df_test2)  

library(caret)
library(gmodels)
df2 <- as.data.frame(lapply(df_test2, unlist))
confusionMatrix(table(prediction ,df2))

