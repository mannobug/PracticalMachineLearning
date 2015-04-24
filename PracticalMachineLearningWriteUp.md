---
title: "PracticalMachineLearning-WriteUp"
output: html_document
---

I have used the data from  http://groupware.les.inf.puc-rio.br/har. The goal is to get a model that be used to predict the Classe variable.

First step was examine data using rattle. During this step I checked that some columns have missing (NA) values. These values are not useful for my analysis and I deicide to remove from my data set. 
Load caret library, load training data set and remove the NA values.


```r
setwd("/Users/emmanuele/Downloads/data")
library(caret)
# Load  data set
trainingAll <- read.csv("pml-training.csv",na.strings=c("NA",""))
# Discard columns with NAs
NAs <- apply(trainingAll, 2, function(x) { sum(is.na(x)) })
trainingValid <- trainingAll[, which(NAs == 0)]
```

Create a subset of the training data set and use only a 20% of the entire set as representative for perform better in the computation with Random Forest Algorithm.


```r
# Create a subset of trainingValid data set
trainIndex <- createDataPartition(y = trainingValid$classe, p=0.2,list=FALSE)
trainData <- trainingValid[trainIndex,]
# Remove useless predictors
removeIndex <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeIndex]
```

After that we can use cross validation with a trainControl with (for example) 4 fold. Finish with this we can apply random forest as follow


```r
# Train control for cross-validation
tc = trainControl(method = "cv", number = 4)

# Random Forests algorithm
modFit <- train(trainData$classe ~.,
                data = trainData,
                method="rf",
                trControl = tc,
                prox = TRUE,
                allowParallel = TRUE)
```
We can check what we've just done simply


```r
print(modFit)
```

```
## Random Forest 
## 
## 3927 samples
##   53 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2945, 2946, 2946, 2944 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9686819  0.9603600  0.005054514  0.006390545
##   27    0.9775947  0.9716544  0.005979350  0.007561040
##   53    0.9714865  0.9639328  0.009842176  0.012431934
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
print(modFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 1.68%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1112   1   1   2   0 0.003584229
## B   14 737   8   0   1 0.030263158
## C    0  13 670   2   0 0.021897810
## D    0   0  13 631   0 0.020186335
## E    1   3   1   6 711 0.015235457
```

With the model in our hand we can use it for predictions on test data set (remove empty values as we did.


```r
# Load test data
testingAll = read.csv("pml-testing.csv",na.strings=c("NA",""))
# Take the columns of testingAll also in trainData
testing <- testingAll[ , which(names(testingAll) %in% names(trainData))]
# Run prediction
pred <- predict(modFit, newdata = testing)
# Utility function discovered in web (previous courses)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred)
```
In conclusion model is really nice and predict 20 cases of 20 even with less data in the model. This approach is very powerful and simplify the solution.
