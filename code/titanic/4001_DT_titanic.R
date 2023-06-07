library(readr)
library(rpart)
library(rpart.plot)
library(data.table)
library(dplyr)
library(plyr)
library(rattle)
library(RColorBrewer)

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



# cross validation with k-fold
k=5
set.seed(123)
cart.model<- rpart(Survived ~. , 
                   data=train,xval=5)

tem=cart.model$variable.importance

df <- data.frame(variable=names(tem),
                 importance=as.numeric(tem))


ggplot(data=df, aes(x=reorder(variable,-importance), y=importance,fill=variable)) +
  geom_bar(stat="identity")+
  ggtitle("Variable importance in bagging model")+
  xlab("Variable") 

fancyRpartPlot(cart.model,sub="Decision Tree raw model")

printcp(cart.model)
plotcp(cart.model)


pred.raw =predict(cart.model, newdata=train, type="class")

cm=confusionMatrix(train$Survived, pred.raw)
cm

# cv-error: misclassfication rate=0.1387

cart.model=rpart(Survived ~. , 
      data=train)
tem=cart.model$variable.importance

df <- data.frame(variable=names(tem),
                 importance=as.numeric(tem))
ggplot(data=df, aes(x=reorder(variable,-importance), y=importance,fill=variable)) +
  geom_bar(stat="identity")+
  ggtitle("Variable importance in bagging model")+
  xlab("Variable") 

ptree<- prune(cart.model, cp=0.010000)
fancyRpartPlot(ptree,sub="Decision Tree prune model")
pred.ptree =predict(ptree, newdata=train, type="class")


cm1=confusionMatrix(train$Survived, pred.ptree)
cm1
# train error: misclassfication rate=0.1387

pred.ptree =predict(ptree, newdata=test, type="class")
cm2=confusionMatrix(pred.ptree,test$Survived)
cm2

pred.ptree=as.numeric(pred.ptree)-1

auc(response=test$Survived, predictor=pred.ptree)
plot.roc(test$Survived,pred.ptree,levels=levels(test$Survived),main="ROC curve of decision tree",
         percent=TRUE,
         col="blue")