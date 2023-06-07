library(gridExtra)
library(caret)
library(ggplot2)
library(dplyr )
library(kernlab)
library(pROC)

# error function
loss_01=function(real,predicted){
  1-mean(real==(predicted>.5))
}

loss_class=function(real,predicted){
  1-mean(real==predicted)
}

loss_sqr=function(real,predicted){
  mean((real-predicted)^2)
}

dat_train=read.csv('Titanic_train_imputed_standardized.csv')
dat_train$Survived=factor(dat_train$Survived)
dat_train_for_fit=dat_train[,c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked')]

dat_test=read.csv('Titanic_test_imputed_standardized.csv')


# train setting
train_ctrl=trainControl(method = 'cv',number = 5)
param_grid_lin=expand.grid(C=10^(-1:2))
param_grid_poly=expand.grid(list(scale=c(1,2,3,4),
                                 C=1,
                                 degree=1:5))
param_grid_radial=expand.grid(list(sigma=c(.5,1,2,3,4),
                                 C=10^(-1:3)))


set.seed(5204001)
svm_Linear = train(Survived ~., 
                    data = dat_train_for_fit, 
                    method = "svmLinear",
                    trControl=train_ctrl,
                   tuneGrid=param_grid_lin)

plt_lin=plot(svm_Linear,main='Linear SVM CV',ylim=c(0,1))


svm_Poly = train(Survived ~., 
                 data = dat_train_for_fit, 
                 method = "svmPoly",
                 trControl=train_ctrl,
                 tuneGrid=param_grid_poly)

plt_poly=plot(svm_Poly,main='Polynomial SVM CV',ylim=c(0,1))

svm_Radial = train(Survived ~., 
                 data = dat_train_for_fit, 
                 method = "svmRadial",
                 trControl=train_ctrl,
                 tuneGrid=param_grid_radial)

plt_radial=plot(svm_Radial,main='Radial SVM CV',ylim=c(0,1))

png('cv_svm.png',width = 1440, height = 720)
grid.arrange(plt_lin,plt_poly,plt_radial,ncol=3)
dev.off()

results=bind_rows(lin=svm_Linear$results,
                     poly=svm_Poly$results,
                     radial=svm_Radial$results,.id = "column_label")
write.csv(results,'cv_svm.csv')
# best params: linear kernel, c = 1
# ksvm_final=ksvm(Survived~.,data = dat_train_for_fit, type = 'C-svc', kernel = 'vanilladot')
# ksvm_final=ksvm(x = as.matrix(dat_train_for_fit[,-1]),
#                 y=as.factor(dat_train_for_fit[,1]), 
#                 kernel = 'vanilladot', 
#                 type = 'C-svc')
# 
# kernlab::plot(ksvm_final,dat_train_for_fit[,3:4])

svm_final=train(Survived ~., 
                  data = dat_train_for_fit, 
                  method = "svmLinear",
                  trControl=trainControl(method = 'none'),
                tuneGrid=expand.grid(C=1))
# kernlab::plot(svm_Linear$finalModel)
pred_train=predict(svm_final,dat_train_for_fit)
pred_class=as.numeric(predict(svm_final,dat_test))-1
real_class=dat_test$Survived

dat_train_pred=dat_train_for_fit
dat_train_pred$Survived=pred_train

plt_pred=ggplot(dat_train_pred,aes(x=Sex,y=Age,colour=Survived))+
  geom_point()+
  ggtitle('Predicted Survived')
plt_real=ggplot(dat_train_for_fit,aes(x=Sex,y=Age,colour=Survived))+
  geom_point()+
  ggtitle('Real Survived')

png('pred_svm.png',width = 960,height = 480)
grid.arrange(plt_pred,plt_real,ncol=2)
dev.off()


confusion_mat=table('real'=real_class,'predicted'=pred_class)
colnames(confusion_mat)=paste('predicted',colnames(confusion_mat))
rownames(confusion_mat)=paste('real',rownames(confusion_mat))
write.csv(confusion_mat,'conf_mat_svm.csv')

auc_svm=auc(response=real_class, predictor=pred_class)

png('roc_svm.png')
plot.roc(real_class, pred_class)
dev.off()


e_output=data.frame('Class Prediction'=c('CV error'=1-svm_Linear$results[1,'Accuracy'],
                                         'Training error'=loss_class(dat_train$Survived,as.numeric(pred_train)-1),
                                         'Testing error'=loss_class(real_class,pred_class),
                                         'AUC'=auc_svm))

write.csv(e_output,'error_svm.csv')
