library(boot)
library(ggplot2)
library(pROC)
# error function
loss_01=function(real,predicted){
  1-mean(real==(predicted>.5))
}

loss_sqr=function(real,predicted){
  mean((real-predicted)^2)
}

bias_var_decomp=function(real, predicted){
  main_guess=mean(predicted)
  c(bias=mean((real-main_guess)^2),
    variance=mean((predicted-main_guess)^2))
}
# logistic regression
dat_train=read.csv('Titanic_train_imputed_standardized.csv')
dat_train_for_fit=dat_train[,c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked')]
dat_test=read.csv('Titanic_test_imputed_standardized.csv')

set.seed(5204001)

# fit model
glm.fit=glm(Survived~.,data=dat_train_for_fit,family=binomial)
# coefficient of  model
write.csv(summary(glm.fit)$coefficients,'leg_reg_coef.csv')

#plotting (not)
variable=names(glm.fit$coefficients)
beta=glm.fit$coefficients
p_val=summary(glm.fit)$coefficients[,4]

ggplot(aes(x=reorder(variable,p_val),y=beta,fill=p_val))+
  geom_bar(stat='identity')+
  geom_text(aes(label=formatC(beta,format="e",digits = 2)), position=position_dodge(width=0.9), vjust=-0.5)+
  ggtitle('Coefficients of Logistic Regression (sorted by significance)')
ggsave(filename = 'logreg_coef.png',device = png())

# cross validation
# logistic regression has no hyper-parameter
# therefore cv is only used to estimate testing error
# but not to find hyper parameter
e_cv_log_01= cv.glm(data=dat_train_for_fit,glmfit=glm.fit,cost=loss_01,K=5)$delta[1]
e_cv_log_sqr= cv.glm(data=dat_train_for_fit,glmfit=glm.fit,cost=loss_sqr,K=5)$delta[1]

# training error
e_train_01=loss_01(predict(glm.fit,dat_train,type='response')>.5,dat_train$Survived)
e_train_sqr=loss_sqr(predict(glm.fit,dat_train,type='response'),dat_train$Survived)

# testing error
pred_prob=predict(glm.fit,dat_test,type='response')
pred_class=as.numeric(predict(glm.fit,dat_test,type='response')>.5)
real_class=dat_test$Survived
n_test=length(real_class)

confusion_mat=table('real'=real_class,'predicted'=pred_class)
colnames(confusion_mat)=paste('predicted',colnames(confusion_mat))
rownames(confusion_mat)=paste('real',rownames(confusion_mat))
write.csv(confusion_mat,'conf_mat_log_reg.csv')


e_test_01=loss_01(pred_class,dat_test$Survived)
e_test_sqr=loss_sqr(pred_prob,dat_test$Survived)

auc_log_reg_class=auc(response=real_class, predictor=pred_class)

png('roc_log_reg_class.png')
plot.roc(real_class, pred_class)
dev.off()

auc_log_reg_prob=auc(response=real_class, predictor=pred_prob)

png('roc_log_reg_prob.png')
plot.roc(real_class, pred_prob)
dev.off()

e_output=data.frame('Class Prediction'=c('CV error'=e_cv_log_01,
                                         'Training error'=e_train_01,
                                         'Testing error'=e_test_01,
                                         'AUC'=auc_log_reg_class),
                    'Prob. Prediction'=c('CV error'=e_cv_log_sqr,
                                         'Training error'=e_train_sqr,
                                         'Testing error'=e_test_sqr,
                                         'AUC'=auc_log_reg_prob),check.names = F)
write.csv(e_output,'error_log_reg.csv')

#bias varaince (not used)
bias_var_prob=bias_var_decomp(predict(glm.fit,dat_test,type='response'),dat_test$Survived)
bias_var_guess=bias_var_decomp(predict(glm.fit,dat_test,type='response')>.5,dat_test$Survived)
