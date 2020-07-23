---
title: 'Practical Machine Learning: Peer Assessment'
output: 
      html_document: 
            keep_md: true
---




## Summary
In this project, we examine weight lifting exercise data collected from various accelerometer
sensors on the belt, forearm, arm and dumbbell of 6 participants. We use this data to build a model
with the primary goal being to predict the type of exercise being performed, given the sensory data.


## Data Transformation
### Loading Required Packages

```r
library(caret)
library(randomForest)
library(rpart)
library(rattle)
set.seed(3233)
```


### Reading Training & Testing Data

```r
trainingOrg <- read.csv("pml-training.csv")
testingOrg <- read.csv("pml-testing.csv")
```


### Choosing Predictors & Making Subsets

```r
subNamesTrain <- grep("^(roll|pitch|yaw|total|gyros|accel|magnet|classe)", names(trainingOrg))
subNamesTest <- grep("^(roll|pitch|yaw|total|gyros|accel|magnet)", names(testingOrg))
training <- trainingOrg[, subNamesTrain]
training$classe <- factor(training$classe)
testing <- testingOrg[, subNamesTest]
```
The subsets are made to remove variables in the dataset that do not help in predicting the exercise
class. These are variables such as mean and standard deviation of the raw sensory data for a single
person and so they are not as useful as the raw sensory data itself. The **grep()** function in the
above code gives the indices of the column names that start with any of the words mentioned in the
function.  
By making the above subsets, the number of total predictors is now **52**.


### Cross Validation
Splitting the training data into two groups (training & testing) to perform cross validation.

```r
indTrain <- createDataPartition(training$classe, p = 0.7, list = F)
trainingCV <- training[indTrain, ]
testingCV <- training[-indTrain, ]
```


## Creating Models
Building two models, decision trees and random forest, then choosing the one with higher accuracy.

### 1. Decision Trees

```r
modelDT <- rpart(factor(classe) ~ ., method = "class", data = trainingCV)
```


```r
fancyRpartPlot(modelDT, sub = "")
```

![](Project_files/figure-html/unnamed-chunk-6-1.png)<!-- -->


```r
cmDT <- confusionMatrix(predict(modelDT, testingCV, type = "class"), testingCV$classe)
cmDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1552  204   11   40   22
##          B   59  691  170  109  159
##          C   34  121  725   77   77
##          D   23   79   81  676  125
##          E    6   44   39   62  699
## 
## Overall Statistics
##                                           
##                Accuracy : 0.738           
##                  95% CI : (0.7265, 0.7492)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6675          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9271   0.6067   0.7066   0.7012   0.6460
## Specificity            0.9342   0.8953   0.9364   0.9374   0.9686
## Pos Pred Value         0.8486   0.5816   0.7012   0.6870   0.8224
## Neg Pred Value         0.9699   0.9046   0.9380   0.9412   0.9239
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2637   0.1174   0.1232   0.1149   0.1188
## Detection Prevalence   0.3108   0.2019   0.1757   0.1672   0.1444
## Balanced Accuracy      0.9307   0.7510   0.8215   0.8193   0.8073
```
As we can see the accuracy of the decision trees model is **0.738**
which is not optimal.


### 2. Random Forest

```r
modelRF <- randomForest(factor(classe) ~ ., method = "class", data = trainingCV)
plot(modelRF, main = "Error for Random Forest Model")
```

![](Project_files/figure-html/unnamed-chunk-8-1.png)<!-- -->


We can see that as the number of trees increases, the error decreases to approximately zero.



```r
cmRF <- confusionMatrix(predict(modelRF, testingCV), testingCV$classe)
cmRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    6    0    0    0
##          B    0 1130    8    0    0
##          C    0    3 1017    6    3
##          D    0    0    1  958    3
##          E    0    0    0    0 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9921   0.9912   0.9938   0.9945
## Specificity            0.9986   0.9983   0.9975   0.9992   1.0000
## Pos Pred Value         0.9964   0.9930   0.9883   0.9958   1.0000
## Neg Pred Value         1.0000   0.9981   0.9981   0.9988   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1920   0.1728   0.1628   0.1828
## Detection Prevalence   0.2855   0.1934   0.1749   0.1635   0.1828
## Balanced Accuracy      0.9993   0.9952   0.9944   0.9965   0.9972
```
The random forest model gives almost perfect accuracy (**0.995**)
on our test set (subset of training set) and so we will choose this model to perform prediction on
the original test set.


## Predicting From Test Data

```r
predict(modelRF, testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
