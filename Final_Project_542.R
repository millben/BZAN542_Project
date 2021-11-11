library(randomForest)
library(rpart)
library(regclass)
library(gbm)
library(caret)
library(pROC)
library(tidyverse)

data = read_csv('VACCINE_DATA.csv')
dim(data)

# discrete option
data.disc = data
data.disc$E.Hesitant = NULL
data.disc['2020'] = NULL
data.disc['State'] = NULL
train.rows = sample(1:nrow(data.disc), 0.8*nrow(data.disc))
train = data.disc[train.rows,]
test = data.disc[-train.rows,]

#LogReg
# mix of lasso a ridge, for each fold, if alpha is closer to 1 its more ridge, closer to 0 more lasso
mod = train(as.factor(Hesitancy.disc)~., data=train, 
            method = "glmnet", 
            metric = "Accuracy",
            tuneLength = 15,
            trControl = trainControl(method="cv",
                                     number = 5,
                                     search = "random",
                                     verboseIter = T))
mod$results[rownames(mod$bestTune),]
ggplot(mod)
varImp(mod)
predicted = predict(mod, newdata=test)
sum(diag(table(predicted, test$Hesitancy.disc)))/nrow(test)



# Random Forest


modelRFC = train(as.factor(Hesitancy.disc) ~., data=train,
                 method="ranger",
                 trControl = trainControl(method = "cv", number=5))
modelRFC
plot(modelRFC)
varImp(modelRFC)


