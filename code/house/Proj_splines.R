######splines######


# load packages and input data
rm(list=ls())
library(caret)
library(ggplot2)
library(gam)
library(tidyverse)

train<-read.csv("~/Downloads/train_sta.csv")
test<-read.csv("~/Downloads/test_sta.csv")
levels(test$LotShape)= c('IR1', 'IR2', 'IR3', 'Reg')

set.seed(123)
# train to find the best span
train_control <- trainControl(method="cv", number=5)
df<-seq(1,10,by=1)
Grid5 <- data.frame(df)
tuned <- train(SalePrice~., data=train[,2:15], trControl=train_control, tuneGrid = Grid5, method="gamSpline")
summary(tuned)
#model[["results"]]

parm <- tuned$bestTune

# final model
mod <- gam(SalePrice ~ LotShape+
             OverallCond+              
             TotRmsAbvGrd + 
             s(WoodDeckSF, df = parm$df)+ 
             s(OpenPorchSF, df = parm$df)+
             s(MasVnrArea, df = parm$df) +
             s(X2ndFlrSF, df = parm$df) +    
             s(LotFrontage, df = parm$df)+
             s(GarageArea, df = parm$df)+
             s(TotalBsmtSF, df = parm$df)+
             s(X1stFlrSF, df = parm$df)+
             s(GrLivArea, df = parm$df)+    
             s(LotArea, df = parm$df), data = train[,2:15]
)
summary(mod)

# pseudo R sq. of the training data
1 - mean((train$SalePrice- mod$fitted.values )^2)/var(train$SalePrice)

# prediction
test<-test[,2:15]
smoothed1 <- predict(mod,newdata = test) 
actual <- test$SalePrice

# residual plot.
res <- test$SalePrice-smoothed1
std_pred_res <- (res-mean(res))/sd(res)

# predict vs actual plot
plot(smoothed1~actual,col='blue',main='prediction vs actual of testing set',ylab='predicted')
abline(lm(smoothed1~actual))

# residual diagnosis plot
hist(std_pred_res,breaks = 20,main='Std residual of prediction Histogram',xlab='std prediction residual',xlim = c(-4,5),probability = T)
lines(density(std_pred_res,adjust = 2),col='red',lwd=2)
plot(smoothed1,std_pred_res, main='std residual vs predicted',xlab='predicted',ylab='std predicted residual')
lines(lowess(smoothed1,std_pred_res),col='red')

# bias, Variance, tested R sq./ MSE
mean((actual - mean(smoothed1))^2) #bias
mean((mean(smoothed1)-smoothed1)^2) #variance
1-sum(res^2)/sum((actual-mean(actual))^2) #tested R sq.


######spline ends######




