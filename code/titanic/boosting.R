library(gridExtra)
library(pROC)
library(ggplot2)
library(caret)
library(pdp)      # for partial dependence plots (PDPs)
library(vip)      # for variable importance plots (VIPs)
# hyper params
n_min=0
n_max=100000
by=50
n_tree=seq(from=n_min, to=n_max, by=by)
shrinkage = 0.001

# 5 fold cv
K=5

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

#get data
dat_train=read.csv('Titanic_train_imputed_standardized.csv')
dat_train_for_fit=dat_train[,c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked')]
dat_train_for_fit$Survived=factor(dat_train_for_fit$Survived,levels = c(0,1))
dat_test=read.csv('Titanic_test_imputed_standardized.csv')
set.seed(5204001)

# e_df=data.frame()

# # cv (not used)
# for(k in 1:K){
#   index_test=dat_train$Folds==k
#   dat_cv_test=dat_train_for_fit[index_test,]
#   dat_cv_train=dat_train_for_fit[!index_test,]
#   boosting_fit_cv=gbm(Survived~.,data=dat_cv_train,n.trees = n_max,shrinkage=shrinkage)
#   predicted=predict(boosting_fit_cv,dat_cv_test,n.trees = n_tree,type="response")
#   e_cv_test=apply(predicted,2,cost2,real=dat_cv_test$Survived)
#   e_df=rbind(e_df,e_cv_test)
#   plot(x=names(e_cv_test),y=e_cv_test)
# }
# 
# e_cv=colMeans(e_df)
# names(e_cv)=n_tree
# png('cv_boosting.png')
# plot(x=n_tree,y=e_cv,type='l',xlab = 'Number of Tree Grown',ylab='Mean Square Error',main = 'Cross Validation Error of Models with Different Parameter')
# dev.off()
# 
# n_trees_best=which.min(e_cv)

# cv 
train_ctrl=trainControl(method = 'cv',number = 5)
param_grid=expand.grid(nrounds=seq(from=250,to=5000,by=250),
                       eta=c(.1,.3,.5),
                         max_depth = 6,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)

set.seed(5204001)
boosting_fit_cv = train(Survived ~., 
                   data = dat_train_for_fit, 
                   method = "xgbTree",
                   trControl=train_ctrl,
                   tuneGrid=param_grid)


results=boosting_fit_cv$results
png('cv_boosting.png')
plot(boosting_fit_cv)
dev.off()

best_param=boosting_fit_cv$best

# outdated
# boosting_fit_training=gbm(Survived~.,data=dat_train_for_fit,n.trees = best_param$nrounds,shrinkage=best_param$eta)
# predicted_test=predict(boosting_fit_training,dat_test[,c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked')],n.trees = n_trees_best,type="response")

# n.trees=2000, shrinkage=1, is the best hyper param here
boosting_fit_training=train(Survived ~., 
                            data = dat_train_for_fit, 
                            method = "xgbTree",
                            trControl=trainControl(method = 'none'),
                            tuneGrid=best_param)
# interpretation
vip(boosting_fit_training)+
  ggtitle('Relative Importance of Variables')
ggsave('rel_inf_boosting.png')

plt_sex=partial(boosting_fit_training,pred.var = 'Sex', which.class = 2, plot.engine = "ggplot2")
plt_sex=autoplot(plt_sex)+ggtitle('Sex Partial Dependence on Survived')
#ggsave('marginal_effect_sex.png')

plt_fare=partial(boosting_fit_training,pred.var = 'Fare',  which.class = 2, plot.engine = "ggplot2")
plt_fare=autoplot(plt_fare)+ggtitle('Fare Partial Dependence on Survived')
#ggsave('marginal_effect_fare.png')

plt_age=partial(boosting_fit_training,pred.var = 'Age',  which.class = 2, plot.engine = "ggplot2")
plt_age=autoplot(plt_age)+ggtitle('Age Partial Dependence on Survived')
#ggsave('marginal_effect_age.png')

plt_pclass=partial(boosting_fit_training,pred.var = 'Pclass',  which.class = 2, plot.engine = "ggplot2")
plt_pclass=autoplot(plt_pclass)+ggtitle('Pclass Partial Dependence on Survived')
#ggsave('marginal_effect_pclass.png')

png('marginal_effect_boosting.png',width = 960, height = 540)
grid.arrange(plt_sex,plt_age,plt_fare,plt_pclass)
dev.off()
# (outdated) interpretation
# rel_imp=summary(boosting_fit_training)
# ggplot(rel_imp,aes(x=reorder(var,-rel.inf),y=rel.inf,fill=rel.inf))+
#   geom_bar(stat='identity')+
#   ggtitle('Relative Influence of Variables ')+
#   ylab('%')+
#   xlab('Variable')+
#   ylim(c(0,100))+
#   theme(legend.position = "none")
# ggsave('rel_inf_boosting.png')
# 
# png('marginal_effect_sex.png')
# plot(boosting_fit_training,i.var = 'Sex')
# dev.off()
# 
# png('marginal_effect_fare.png')
# plot(boosting_fit_training,i.var = 'Fare')
# dev.off()
# 
# png('marginal_effect_age.png')
# plot(boosting_fit_training,i.var = 'Age')
# dev.off()
# 
# png('marginal_effect_pclass.png')
# plot(boosting_fit_training,i.var = 'Pclass')
# dev.off()


# cost2(dat_test$Survived,predicted_test)
# cost(dat_test$Survived,predicted_test>.5)

#prediction
pred_train=apply(predict(boosting_fit_training,dat_train,type='prob'),1,which.max)-1

pred_prob=predict(boosting_fit_training,dat_test,type='prob')
pred_class=apply(pred_prob,1,which.max)-1
real_class=dat_test$Survived
n_test=length(real_class)

confusion_mat=table('real'=real_class,'predicted'=pred_class)
colnames(confusion_mat)=paste('predicted',colnames(confusion_mat))
rownames(confusion_mat)=paste('real',rownames(confusion_mat))
write.csv(confusion_mat,'conf_mat_boosting.csv')

auc_boosting_class=auc(response=real_class, predictor=pred_class)

png('roc_boosting_class.png')
plot.roc(real_class, pred_class)
dev.off()


e_output=data.frame('Class Prediction'=c('CV error'=1-max(boosting_fit_cv$results[,'Accuracy']),
                                         'Training error'=loss_class(dat_train$Survived,pred_train),
                                         'Testing error'=loss_class(real_class,pred_class),
                                         'AUC'=auc_boosting_class))

write.csv(e_output,'error_boosting.csv')
