#------------------------- Packages ---------------------
install.packages("car")
install.packages("psycho")
library(dplyr)
library(tidyverse)
library(psycho)
library(caret)
library(car)
library(MASS)



#---------------------------- Data preparation ----------------
set.seed(123)

##### Data Load
train <- read.csv('~/Desktop/STAT4001/train_sta.csv',header = T)[-1]
test <- read.csv('~/Desktop/STAT4001/test_sta.csv',header = T)[-1]
levels(test$LotShape) <- c('IR1', 'IR2', 'IR3', 'Reg')
levels(test$OverallCond) <- c('1','2','3','4','5','6','7','8','9')

#####Change the internal structure of the variables
cat_vars <- c("LotShape","OverallCond")
train[, cat_vars] <- lapply(train[, cat_vars], as.factor)
test[, cat_vars] <- lapply(test[, cat_vars], as.factor)

###### Check the Linear regression assumptions 
mul_reg<-lm(SalePrice~.,data=train)
par(mfrow=c(2,2))
plot(mul_reg) 


#---------------- Model 1: Perform cross-validation with Stepwise Selection --------------
set.seed(4001)

###########  Define training control
train.control <- trainControl(method = "cv", number = 5)
###########  Train the model
model_cv_seq <- train(SalePrice~., data = train, method = "leapSeq",metric='RMSE',
                  trControl = train.control)
###Summary of model
print(model_cv_seq)


########### Make predictions 
predictions_cv_seq_test <- model_cv_seq %>% predict(test)
predictions_cv_seq_train <- model_cv_seq %>% predict(train)

########### Residual Plot
par(mfrow=c(1,1))
res = resid(model_cv_seq)
plot(train$SalePrice, res, 
     ylab="Residuals", xlab="Sale Price", 
     main="Residual Plot of CV Stepwise Model") 
abline(0, 0)  
pred <- predict.train(model_cv_seq,newdata = test)
actual <- test$SalePrice
res <- actual-pred
std_pred_res <- (res-mean(res))/sd(res)
par(mfrow=c(1,2))
hist(std_pred_res,breaks = 20,main='Std residual of prediction Histogram',xlab='std prediction residual',xlim = c(-4,5),probability = T)
lines(density(std_pred_res,adjust = 2),col='red',lwd=2)
plot(pred,std_pred_res, main='std residual vs predicted',xlab='predicted',ylab='std predicted residual')
lines(lowess(pred,std_pred_res),col='red')

par(mfrow=c(1,1))
##### Plot Importance of variables
ggplot(varImp(model_cv_seq))

########### Find Summary of Model 1
#1)Tested R-Square
1-sum((test$SalePrice-predictions_cv_seq_test)^2)/sum((test$SalePrice-mean(test$SalePrice))^2)
#2)Bias 
bias.est<-test$SalePrice-mean(predictions_cv_seq_test)
(bias<-mean(bias.est^2))
#3)Variance
(variance<-mean((mean(predictions_cv_seq_test)-predictions_cv_seq_test)^2))
#4)Training Error
(1/dim(train)[1])*sum((train$SalePrice-predictions_cv_seq_train)^2)
#5)Testing Error
(1/dim(test)[1])*sum((test$SalePrice-predictions_cv_seq_test)^2)
#6)CV Error
mean((model_cv_seq$results$RMSE)^2)




#---------------- Model 2: Perform cross-validation with step backward --------------
set.seed(4011)
###########  Define training control
train.control <- trainControl(method = "cv", number = 5)

###########  Train the model
model_cv_stepback <- train(SalePrice ~., data = train,
                    method = "leapBackward" ,metric='RMSE',
                    tuneGrid = data.frame(nvmax = 1:12),
                    trControl = train.control
)
model_cv_stepback$results
model_cv_stepback$bestTune #12
summary(model_cv_stepback$finalModel)
coef(model_cv_stepback$finalModel, 12)
model_cv_stepback<-lm(SalePrice ~ OverallCond+ TotalBsmtSF+`1stFlrSF` +`2ndFlrSF`+GarageArea , 
   data = train)
###Summary of model
print(model_cv_stepback)

########### Residual Plot
par(mfrow=c(1,1))
res = resid(model_cv_stepback)
plot(train$SalePrice, res, 
     ylab="Residuals", xlab="Sale Price", 
     main="Residual Plot of CV Backwards Selection Model") 
abline(0, 0)  
pred <- predict.train(model_cv_stepback,newdata = test)
actual <- test$SalePrice
res <- actual-pred
std_pred_res <- (res-mean(res))/sd(res)
par(mfrow=c(1,2))
hist(std_pred_res,breaks = 20,main='Std residual of prediction Histogram',xlab='std prediction residual',xlim = c(-4,5),probability = T)
lines(density(std_pred_res,adjust = 2),col='red',lwd=2)
plot(pred,std_pred_res, main='std residual vs predicted',xlab='predicted',ylab='std predicted residual')
lines(lowess(pred,std_pred_res),col='red')

########### Make predictions 
predictions_cv_stepback_test <- model_cv_stepback %>% predict(test)
predictions_cv_stepback_train <- model_cv_stepback %>% predict(train)

par(mfrow=c(1,1))

##### Plot Importance of variables
ggplot(varImp(model_cv_stepback))

########### Find Summary of Model 2
#1)Tested R-Square
1-sum((test$SalePrice-predictions_cv_stepback_test)^2)/sum((test$SalePrice-mean(test$SalePrice))^2)
#2)Bias 
bias.est<-test$SalePrice-mean(predictions_cv_stepback_test)
(bias<-mean(bias.est^2))
#3)Variance
(variance<-mean((mean(predictions_cv_stepback_test)-predictions_cv_stepback_test)^2))
#4)Training Error
(1/dim(train)[1])*sum((train$SalePrice-predictions_cv_stepback_train)^2)
#5)Testing Error
(1/dim(test)[1])*sum((test$SalePrice-predictions_cv_stepback_test)^2)
#6)CV Error
mean((model_cv_stepback$results$RMSE)^2)


#---------------- Model 3: Perform cross-validation with step forward --------------
set.seed(4011)
###########  Define training control
train.control <- trainControl(method = "cv", number = 5)

###########  Train the model
model_cv_stepfor <- train(SalePrice ~., data = train,
                           method = "leapForward" ,metric='RMSE',
                           tuneGrid = data.frame(nvmax = 1:12),
                           trControl = train.control
)
print(model_cv_stepfor)



model_cv_stepfor$results
model_cv_stepfor$bestTune #12
summary(model_cv_stepfor$finalModel)
coef(model_cv_stepfor$finalModel, 12)
model_cv_stepfor<-lm(SalePrice ~ LotShapeReg+OverallCond+ TotalBsmtSF+X2ndFlrSF+
                       GrLivArea+TotRmsAbvGrd+GarageArea+WoodDeckSF, 
                      data = train)
###Summary of model
print(model_cv_stepfor)

########### Residual Plot
par(mfrow=c(1,1))
res = resid(model_cv_stepfor)
plot(train$SalePrice, res, 
     ylab="Residuals", xlab="Sale Price", 
     main="Residual Plot of CV Forward Selection Model") 
abline(0, 0) 
pred <- predict.train(model_cv_stepfor,newdata = test)
actual <- test$SalePrice
res <- actual-pred
std_pred_res <- (res-mean(res))/sd(res)
par(mfrow=c(1,2))
hist(std_pred_res,breaks = 20,main='Std residual of prediction Histogram',xlab='std prediction residual',xlim = c(-4,5),probability = T)
lines(density(std_pred_res,adjust = 2),col='red',lwd=2)
plot(pred,std_pred_res, main='std residual vs predicted',xlab='predicted',ylab='std predicted residual')
lines(lowess(pred,std_pred_res),col='red')

########### Make predictions 
predictions_cv_stepfor_test <- model_cv_stepfor %>% predict(test)
predictions_cv_stepfor_train <- model_cv_stepfor %>% predict(train)

par(mfrow=c(1,1))

##### Plot Importance of variables
ggplot(varImp(model_cv_stepfor))


########### Find Summary of Model 3
#1)Tested R-Square
1-sum((test$SalePrice-predictions_cv_stepfor_test)^2)/sum((test$SalePrice-mean(test$SalePrice))^2)
#2)Bias 
bias.est<-test$SalePrice-mean(predictions_cv_stepfor_test)
(bias<-mean(bias.est^2))
#3)Variance
(variance<-mean((mean(predictions_cv_stepfor_test)-predictions_cv_stepfor_test)^2))
#4)Training Error
(1/dim(train)[1])*sum((train$SalePrice-predictions_cv_stepfor_train)^2)
#5)Testing Error
(1/dim(test)[1])*sum((test$SalePrice-predictions_cv_stepfor_test)^2)
#6)CV Error
mean((model_cv_stepfor$results$RMSE)^2)








