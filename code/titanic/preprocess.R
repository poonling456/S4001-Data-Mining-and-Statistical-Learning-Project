# read data
setwd("C:/Users/PoonLing/The Chinese University of Hong Kong/FUNG, Chun Yin - 4001_proj/titanic")
raw_full=read.csv('Titanic.csv',na.strings = '')

# clean data
# "If the age is estimated, is it in the form of xx.5"
# separate age in raw data into 2 variable, 
# 1) age and 2) is the age estimated
age_is_estimated=raw_full$Age%%1==.5 & raw_full$Age>1
age_is_estimated[is.na(age_is_estimated)]=F
data_clean=raw_full
data_clean[age_is_estimated,'Age']=data_clean[age_is_estimated,'Age']-.5
data_clean$IsAgeEstimated=ifelse(age_is_estimated,T,F)

# basic info of the data
n_total=nrow(raw_full)
proportion_test=1/4
predictor=c('Pclass','Sex','Age','SibSp','Parch','Fare','Embarked')

# separate test data set and train data set
seed=5204001
set.seed(seed)
test_index=sample(1:n_total,n_total/4,replace = F)
raw_test=raw_full[test_index,]
write.csv(raw_test,'Titanic_test.csv',row.names=F)

raw_train=raw_full[-test_index,]
write.csv(raw_test,'Titanic_train.csv',row.names=F)

# impute missing values
library(missForest)
imputed_test=raw_test
imputed_test[,predictor]=missForest(raw_test[,predictor])$ximp
write.csv(imputed_test,'Titanic_test_imputed.csv',row.names=F)

imputed_train=raw_train
imputed_train[,predictor]=missForest(raw_train[,predictor])$ximp

# assign cross validation fold 
library(caret)
imputed_train$Folds= createFolds(imputed_train$Survived, k = 5,list = F)
write.csv(imputed_train,'Titanic_train_imputed.csv',row.names=F)

# standardize training data
train_imp_std=imputed_train

#min max standization
sibsp_rng=range(train_imp_std$SibSp)
train_imp_std$SibSp=(train_imp_std$SibSp-sibsp_rng[1])/(sibsp_rng[2]-sibsp_rng[1])

parch_rng=range(train_imp_std$Parch)
train_imp_std$Parch=(train_imp_std$Parch-parch_rng[1])/(parch_rng[2]-parch_rng[1])

pclass_rng=range(train_imp_std$Pclass)
train_imp_std$Pclass=(train_imp_std$Pclass-pclass_rng[1])/(pclass_rng[2]-pclass_rng[1])

#z-score standardize
age_mean=mean(train_imp_std$Age)
age_sd=sd(train_imp_std$Age)

train_imp_std$Age=(train_imp_std$Age-age_mean)/age_sd


fare_mean=mean(train_imp_std$Fare)
fare_sd=sd(train_imp_std$Fare)

train_imp_std$Fare=(train_imp_std$Fare-fare_mean)/fare_sd
write.csv(imputed_train,'Titanic_train_imputed_standardized.csv',row.names=F)



# standardize testing data
test_imp_std=imputed_test

#min max standization
test_imp_std$SibSp=(test_imp_std$SibSp-sibsp_rng[1])/(sibsp_rng[2]-sibsp_rng[1])

test_imp_std$Parch=(test_imp_std$Parch-parch_rng[1])/(parch_rng[2]-parch_rng[1])

test_imp_std$Pclass=(test_imp_std$Pclass-pclass_rng[1])/(pclass_rng[2]-pclass_rng[1])

#z-score standardize
test_imp_std$Age=(test_imp_std$Age-age_mean)/age_sd

test_imp_std$Fare=(test_imp_std$Fare-fare_mean)/fare_sd

write.csv(imputed_test,'Titanic_test_imputed_standardized.csv',row.names=F)
