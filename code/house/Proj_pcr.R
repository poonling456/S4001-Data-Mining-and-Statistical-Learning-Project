######pcr######


# load packages and input data
rm(list=ls())
library(pls)
library(ggplot2)

train<-read.csv("~/Downloads/train_sta.csv")
test<-read.csv("~/Downloads/test_sta.csv")
levels(test$LotShape)= c('IR1', 'IR2', 'IR3', 'Reg')

# training model
set.seed (123)
pcr_model <- pcr(SalePrice~., data = train[,2:15], scale = TRUE, validation = "CV",segments = 5)
summary(pcr_model)

# plot the root mean squared error
validationplot(pcr_model,val.type="MSEP")
n <- selectNcomp(pcr_model,method = "randomization")
cverr <- RMSEP(pcr_model)$val[1,,]
plot(c(0:15),cverr,main = 'cv error vs number of components',xlab='number of components',type = "l", lty=2)

# final model
fmodel <- pcr(SalePrice~.,data=train[,2:15],ncomps=n)

# visualization
coefplot(fmodel,ncomp = n,type='h')
predplot(fmodel,newdata = test,which = 'test',line=T,col='blue',xlab='actual',main='prediction vs actual of testing set')

# pseudo R sq. of the training data
1-mean((train$SalePrice-fmodel$fitted.values)^2)/var(train$SalePrice)


# prediction
pred <- predict(fmodel, ncomp = n, newdata = test[,2:15], type=c('response','scores'))
actual <- test$SalePrice

# residual plot.
res <- test$SalePrice-pred
std_pred_res <- (res-mean(res))/sd(res)

# predict vs actual plot
plot(pred~actual,col='blue',main='prediction vs actual of testing set',ylab='predicted')
abline(lm(pred~actual))

# residual diagnosis plot
hist(std_pred_res,breaks = 20,main='Std residual of prediction Histogram',xlab='std prediction residual',xlim = c(-4,5),probability = T)
lines(density(std_pred_res,adjust = 2),col='red',lwd=2)
plot(pred,std_pred_res, main='std residual vs predicted',xlab='predicted',ylab='std predicted residual')
lines(lowess(pred,std_pred_res),col='red')

# bias, variance, tested R sq./ MSE
mean((actual - mean(pred))^2) #bias
mean((mean(pred)-pred)^2) #variance
1-sum(res^2)/sum((actual-mean(actual))^2) #tested R sq.


######pcr ends######





