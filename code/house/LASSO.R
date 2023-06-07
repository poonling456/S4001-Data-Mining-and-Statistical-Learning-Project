install.packages("glmnet")
library(tidyverse)
library(glmnet)
library(dplyr)
library(caret)


#---------------------------- Data preparation ----------------
is.imp <- TRUE
########### Data Load
if(is.imp){
  train <- read.csv('~/Desktop/STAT4001/train_sta.csv',header = T)[,-1]
  test <- read.csv('~/Desktop/STAT4001/test_sta.csv',header = T)[,-1]
}else{
  train <- read.csv('~/Desktop/STAT4001/train_house.csv',header = T)[,-1]
  test <- read.csv('~/Desktop/STAT4001/test_house.csv',header = T)[,-1]
}
#####Change the internal structure of the variables
cat_vars <- c("LotShape","OverallCond")
train[, cat_vars] <- lapply(train[, cat_vars], as.factor)
test[, cat_vars] <- lapply(test[, cat_vars], as.factor)

levels(test$LotShape) <- c('IR1', 'IR2', 'IR3', 'Reg')
levels(test$OverallCond) <- c('1','2','3','4','5','6','7','8','9')
house<- rbind(train,test)



house_Y<-house$SalePrice
house_X<-model.matrix(SalePrice~.,house)[,-1]
y_train<-train$SalePrice
x_train<-model.matrix(SalePrice~.,train)[,-1]
y_test<-test$SalePrice
x_test<-model.matrix(SalePrice~.,test)[,-1]


#---------------------------- LASSO Regression ----------------
set.seed(123)

######## Perform lasso regression
grid<-10^seq(10,-2,length=100)
mod_l<-cv.glmnet(x_train,y_train,alpha=1,
               lambda = grid, 
               nfolds = 5,
               standaradize = FALSE
)

######## CV error vs lamda plot
plot(mod_l)
######## best lamda
(bestlam2<-mod_l$lambda.min)
######## which one is the best in the grid
which(mod_l$lambda.min==grid)
######## check whether the corresponding CV error is the smallest among all
mod_l$cvm[which(mod_l$lambda.min==grid)]==min(mod_l$cvm)


par(mfrow=c(1,1))
########### Refit lasso regression model on the full data set using lambda chosen from cross-validation
mod<-glmnet(house_X,house_Y,alpha=1,lambda=grid)
lasso.coef=predict(mod,type="coefficients",s=bestlam2)[1:22,]
lasso.coef
lasso.coef[lasso.coef!=0]

########### Residual Plot and prediction plots
res = resid(mod)
plot(train$SalePrice, res, 
     ylab="Residuals", xlab="Sale Price", 
     main="Residual Plot of LASSO Model") 
abline(0, 0) 
pred <- predict(mod,s=bestlam2,newx=x_test)
actual <- test$SalePrice
##prediction vs actual of testing set
plot(pred~actual,col='blue',main='prediction vs actual of testing set',ylab='predicted')
abline(lm(pred~actual))
##Std residual of prediction Histogram
res <- actual-pred
std_pred_res <- (res-mean(res))/sd(res)
hist(std_pred_res,breaks = 20,main='Std residual of prediction Histogram',xlab='std prediction residual',xlim = c(-4,5),probability = T)
lines(density(std_pred_res,adjust = 2),col='red',lwd=2)
##std residual vs predicted
plot(pred,std_pred_res, main='std residual vs predicted',xlab='predicted',ylab='std predicted residual')
lines(lowess(pred,std_pred_res),col='red')


########### Make predictions 
predictions_test <- predict(mod,s=bestlam2,newx=x_test)
predictions_train <- predict(mod,s=bestlam2,newx=x_train)

########### Summary of LASSO Model 
#1)Tested R-Square
1-sum((test$SalePrice-predictions_test)^2)/sum((test$SalePrice-mean(test$SalePrice))^2)
#2)Bias 
bias.est<-test$SalePrice-mean(predictions_test)
(bias<-mean(bias.est^2))
#3)Variance
(variance<-mean((mean(predictions_test)-predictions_test)^2))
#4)Training Error
(1/dim(train)[1])*sum((train$SalePrice-predictions_train)^2)
#5)Testing Error
(1/dim(test)[1])*sum((test$SalePrice-predictions_test)^2)
#6)CV error
min(mod_l$cvm)
