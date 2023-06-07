######leoss######


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
degree<-rep_len(1,99)
span<-seq(0.01,0.99,by=0.01)
Grid5 <- data.frame(span,degree)
model <- train(SalePrice~., data=train[,2:15], trControl=train_control, tuneGrid = Grid5, method="gamLoess")
summary(model)
model$bestTune
#model[["results"]]
# the best span is 0.99

# final model
set.seed(123)
parm <- model$bestTune

# final model
mod <- gam(SalePrice ~ LotShape +
             OverallCond+
             TotRmsAbvGrd+
             lo(WoodDeckSF, span = parm$span, degree = parm$degree)+
             lo(OpenPorchSF, span = parm$span, degree = parm$degree)+
             lo(MasVnrArea, span = parm$span, degree = parm$degree)+
             lo(X2ndFlrSF, span = parm$span, degree = parm$degree)+ 
             lo(LotFrontage, span = parm$span, degree = parm$degree)+
             lo(GarageArea, span = parm$span, degree = parm$degree)+
             lo(TotalBsmtSF, span = parm$span, degree = parm$degree)+
             lo(X1stFlrSF, span = parm$span, degree = parm$degree)+
             lo(GrLivArea, span = parm$span, degree = parm$degree)+   
             lo(LotArea, span = parm$span, degree = parm$degree), data = train[,2:15]
)
summary(mod)

# pseudo R sq. of the training data
1 - mean((train$SalePrice- mod$fitted.values )^2)/var(train$SalePrice)

# prediction
smoothed1 <- predict(mod,newdata = test[,2:15]) 
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


######leoss ends######







