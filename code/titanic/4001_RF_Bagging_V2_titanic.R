#################
# 1. Data import
#################
set.seed(123)
library(readr)
library(dplyr)
library(randomForest)
library(caret)

train=read_csv("D:/STAT4001/Titanic_train_imputed_standardized.csv")
test=read_csv("D:/STAT4001/Titanic_test_imputed_standardized.csv")

train$PassengerId=NULL
train$Folds=NULL
test$PassengerId=NULL
sapply(train,class)
sapply(test,class)

dim(train)
dim(test)

# Train -- changing the class of variables

train$Survived=as.factor(train$Survived)
train$Pclass=as.factor(train$Pclass)
train$Sex=as.factor(train$Sex)
train$Embarked=as.factor(train$Embarked)
sapply(train,class)

# test -- changing the class of variables

test$Survived=as.factor(test$Survived)
test$Pclass=as.factor(test$Pclass)
test$Sex=as.factor(test$Sex)
test$Embarked=as.factor(test$Embarked)
sapply(test,class)


# CV Source: https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/


metric <- "Accuracy"
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# train model


control <- trainControl(method="cv", number=5)
tunegrid <- expand.grid(.mtry=c(1:9), .ntree=c(500,1000,1500,2000,2500))
set.seed(123)
custom <- train(Survived~., data=train, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control,na.action = na.omit)
custom$results
print(custom)
plot(custom)
cv_error=custom$results

library(ggplot2)
cv_error=custom$results[,1:3]
cv_error[,3]=1-cv_error[,3]
names(cv_error)=c("mtry","ntree","CV_error")
cv_error$ntree=as.factor(cv_error$ntree)

ggplot(cv_error, aes(x=mtry,y=CV_error,col=ntree))+ 
  geom_line()+
  scale_x_discrete(limits=c(0:9))+
  labs(title="Comparsion on CV error rate between different RF models")

min(cv_error[,3])
# the cv error is 0.1520465 for best model mtry=2,ntree=1500

custom$finalModel

custom$bestTune
mtry=as.numeric(custom$bestTune[1])
ntree=as.numeric(custom$bestTune[2])


# training using all train dataset
# reference: https://towardsdatascience.com/random-forest-in-r-f66adf80ec9
rf = randomForest(Survived ~ .,data=train,mtry=mtry,ntree=ntree)
tem1=importance(rf)
df <- data.frame(variable=rownames(tem1),
                 Gini=as.numeric(tem1[,1]))

# https://sebastiansauer.github.io/ordering-bars/

ggplot(data=df, aes(x=reorder(variable,-Gini), y=Gini,fill=variable)) +
  geom_bar(stat="identity")+
  ggtitle("Variable importance in rf model")+
  xlab("Variable") 

# predict in train set
pred_train = predict(rf, newdata=train[,-1])
confusionMatrix(train$Survived,pred_train)

# misclassfication rate=0.064

# predict in test set
pred_test = predict(rf, newdata=test[,-1])
cm1=confusionMatrix(test$Survived, pred_test)
cm1

pred_test=as.numeric(pred_test)-1
auc(response=test$Survived, predictor=pred_test)
plot.roc(test$Survived,pred_test,levels=levels(test$Survived),main="ROC curve of RF",
         percent=TRUE,
         col="blue")


pred_test_prob = predict(rf, newdata=test[,-1],type = "prob")
dt_pred=pred_test_prob[,2]
test$Survived=as.numeric(test$Survived)
error_rate=mean((dt_pred-test$Survived)^2) 
error_rate

# error_rate 1.315797


# BAGGING
# http://rstudio-pubs-static.s3.amazonaws.com/156481_80ee6ee3a0414fd38f5d3ad33d14c771.html
# bagging in random forest

tunegrid <- expand.grid(.mtry=9, .ntree=c(500,1000,1500,2000,2500))
set.seed(123)
custom.bag <- train(Survived~., data=train, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
print(custom.bag)
plot(custom.bag)

custom.bag

cv_error=custom.bag$results[,1:3]
cv_error[,3]=1-cv_error[,3]
min(cv_error[,3])
names(cv_error)=c("mtry","ntree","CV_error")

ggplot(cv_error, aes(x=ntree,y=CV_error))+ 
  geom_line(color = "blue")+
  labs(title="Comparsion on CV error rate between different bagging models")

custom.bag$bestTune
ntree=as.numeric(custom$bestTune[2])

# training using all train dataset
# reference: https://towardsdatascience.com/random-forest-in-r-f66adf80ec9
rf.bag = randomForest(Survived ~ .,data=train,mtry=7,ntree=ntree)
tem2=importance(rf.bag)

df <- data.frame(variable=rownames(tem2),
                 Gini=as.numeric(tem2[,1]))

ggplot(data=df, aes(x=reorder(variable,-Gini), y=Gini,fill=variable)) +
  geom_bar(stat="identity")+
  ggtitle("Variable importance in bagging model")+
  xlab("Variable") 

# predict in train set

pred_train = predict(rf.bag, newdata=train[,-1])
confusionMatrix(train$Survived,pred_train)

# predict in test set
pred_test = predict(rf.bag, newdata=test[,-1])
cm2=confusionMatrix(test$Survived,pred_test)
cm2

pred_test=as.numeric(pred_test)-1
auc(response=test$Survived, predictor=pred_test)
plot.roc(test$Survived,pred_test,levels=levels(test$Survived),main="ROC curve of bagging",
         percent=TRUE,
         col="blue")




pred_test_prob = predict(rf.bag, newdata=test[,-1],type = "prob")
dt_pred=pred_test_prob[,2]
test$Survived=as.numeric(test$Survived)
error_rate=mean((dt_pred-test$Survived)^2) 
error_rate

# error_rate is 1.243189
