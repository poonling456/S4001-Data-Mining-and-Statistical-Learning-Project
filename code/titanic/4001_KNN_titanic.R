library(kknn)
library(readr)
library(rpart)
library(rpart.plot)
library(data.table)
library(dplyr)
library(plyr)
library(class)

st.seed=115

tit=read_csv("D:/STAT4001/Titanic.csv")


id=tit$PassengerId
tit$PassengerId=NULL


count(tit$Age)
tit$Age[which(is.na(tit$Age))]=mean(tit$Age,na.rm = T)


count(tit$Embarked)
tit$L_embarked=rep(0,500)
dummy=which(is.na(tit$Embarked))
tit$Embarked[dummy]=0

for(i in 1:500){
  if(tit$Embarked[i]=="Q") tit$L_embarked[i]=1
  else if((tit$Embarked[i]=="S")) tit$L_embarked[i]=2
  else tit$L_embarked[i]=0
  
}

tit$Embarked=NULL


# tit$Survived=as.factor(tit$Survived)
# tit$Pclass=as.factor(tit$Pclass)
# tit$Sex=as.factor(tit$Sex)
# tit$Embarked=NULL
# tit$L_embarked=as.factor(tit$L_embarked)


tit$Sex=as.character(tit$Sex)
tit$Sex[which(tit$Sex=="male")]=0
tit$Sex[which(tit$Sex=="female")]=1
tit$Sex=as.numeric(tit$Sex)
count(tit$Sex)



tit_train=tit[1:400,]
tit_test=tit[401:500,]

x=tit_train$Survived
tit_train$Survived=NULL

y=tit_test$Survived
tit_test$Survived=NULL


class(tit$Survived)
class(tit$Pclass)
class(tit$Sex)
class(tit$Age)
class(tit$SibSp)
class(tit$Parch)
class(tit$Fare)
class(tit$L_embarked)

#knn require no NA!
knn=knn(tit_train, tit_test, x, k = 1, l = 0) 
#use.all controls handling of ties. If true, all distances equal to the kth largest are included. 
#If false, a random selection of distances equal to the kth is chosen to use exactly k neighbours.
tab=table(knn,y)
tab

accuracy <- function(w){sum(diag(w)/(sum(rowSums(w)))) * 100}
accuracy(tab)
