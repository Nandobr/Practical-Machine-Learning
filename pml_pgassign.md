Executive Summary
-----------------

One thing that people regularly do is quantify how much of a particular
activity they do, but they rarely quantify how well they do it. In this
project, the goal is to use data from accelerometers on the belt,
forearm, arm, and dumbell of 6 participants.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. They were asked to
perform barbell lifts correctly and incorrectly in 5 different ways.
More information is available from the website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
(see the section on the Weight Lifting Exercise Dataset).

About the Data :
----------------

The training data for this project are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.
If you use the document you create for this class for any purpose please
cite them as they have been very generous in allowing their data to be
used for this kind of assignment.

Loading the data
----------------

    # set working diretory
    setwd("~/R_Coursera/Practical Machine Learning")
    # Download training dataset if it is not already in the directory 
    if (!file.exists("har.txt")) {
          url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
          download.file(url,"har.txt")
          }
    # Download testing dataset if it is not already in the directory 
    if (!file.exists("har2.txt")) {
          url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
          download.file(url2,"har2.txt")
          }
    # Load datasets
    train <- read.csv("har.txt",sep = ",", header = TRUE,na.strings= c('#DIV/0', '', 'NA'))
    test <- read.csv("har2.txt",sep = ",", header = TRUE,na.strings= c('#DIV/0', '', 'NA'))

Data Exploration
----------------

Let's take a look at the dimensions of the training and testing
datasets, and also take a look at the outcome variable CLASSE to see how
it is distributed on this sample data.

It is important to check if there are NA values on training dataset,
because they may cause errors during future procedures.

    # Check dataset dimensions
    dim(train)

    ## [1] 19622   160

    dim(test)

    ## [1]  20 160

    # check outcome(classe) distribution
    summary(train$classe)

    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

    # check NA values
    sum(is.na(train))

    ## [1] 1921600

    sum(is.na(test))

    ## [1] 2000

Data Transformation
-------------------

Note that there many columns with NA values and they need to be
discarded. Also, exploring the content of the dataset, we can observe
that the seven intial columns have no correlation with the outcome
variable prediction and they will be discarded as well.

    # Process training dataset
    # Discard columns with NA values and not correlated ones
    Nac <- sapply(1:160,function(n){sum(is.na(train[,n]))})
    cwithNA <- which(Nac>0)
    train <- train[,-cwithNA]
    train <- train[,-c(1:7)]
    # Process test dataset
    test <- test[,-cwithNA]
    test <- test[,-c(1:7)]

Now, after removing NA values, we have train and test datasets with less
columns and smaller sizes, which will be less computer demanding to be
processed. New dimensions are:

    # Check new datasets dimensions
    dim(train)

    ## [1] 19622    53

    dim(test)

    ## [1] 20 53

Cross Validation
----------------

Instead of doing a single random data partition, we're gonna use cross
validation. A 10-fold cross validation will be used with the
traincontrol() function of the caret package.

Below we're going to to fit Random Forest(rf) and Linear discriminant
analysis(lda) models to see how they perform.

    # Set the seed for reproducibility
    set.seed(1234)
    # Load necessary libraries 
    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(randomForest)

    ## Warning: package 'randomForest' was built under R version 3.4.2

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    # Cross validation - 10 folds
    cv10 <- trainControl(method = "cv",allowParallel = TRUE, number = 10)

Random Forest Model Prediction
------------------------------

    # Since it takes more than an hour to fit RF model, I'll check if it already exists.
    # Start the clock!
    # ptm <- proc.time()
    # fit Random Forest(RF) model
    # fitRf <- train(classe~.,data=train, method="rf",allowParallel=TRUE,trcontorl=cv10)
    # Stop the clock
    # (proc.time() - ptm)/60
    # Time spent(in minutes) to fit the model : 73 minutes
    # usuário    sistema  decorrido 
    # 62.1833333  0.2756667 73.0075000 
    # load saved fitted model from disk
    fitRf <- readRDS("./fitRf_model.rds")
    # predicting the outcome variable(classe) on training dataset
    predRf <- predict(fitRf,train)
    table(predRf,train$classe)

    ##       
    ## predRf    A    B    C    D    E
    ##      A 5580    0    0    0    0
    ##      B    0 3797    0    0    0
    ##      C    0    0 3422    0    0
    ##      D    0    0    0 3216    0
    ##      E    0    0    0    0 3607

    confusionMatrix(predRf,train$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 5580    0    0    0    0
    ##          B    0 3797    0    0    0
    ##          C    0    0 3422    0    0
    ##          D    0    0    0 3216    0
    ##          E    0    0    0    0 3607
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9998, 1)
    ##     No Information Rate : 0.2844     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Linear discriminant analysis Prediction
---------------------------------------

    # fit Linear discriminant analysis (LDA) model
    # Start the clock!
    ptm <- proc.time()
    fitLda <- train(classe~.,data=train,method="lda",trcontorl=cv10)
    # Stop the clock
    (proc.time() - ptm)/60

    ##       user     system    elapsed 
    ## 0.13466667 0.01366667 0.14883333

    # Time spent(in minutes) to fit the model: 0.3 minute
    # user     system    elapsed 
    # 0.19666667 0.01416667 0.28716667 

    # Predicting with Linear discriminant analysis(LDA) model
    predLda <- predict(fitLda,train)
    table(predLda,train$classe)

    ##        
    ## predLda    A    B    C    D    E
    ##       A 4568  586  341  191  133
    ##       B  121 2429  333  130  611
    ##       C  444  455 2254  379  323
    ##       D  429  148  411 2383  344
    ##       E   18  179   83  133 2196

    confusionMatrix(predLda,train$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4568  586  341  191  133
    ##          B  121 2429  333  130  611
    ##          C  444  455 2254  379  323
    ##          D  429  148  411 2383  344
    ##          E   18  179   83  133 2196
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7048          
    ##                  95% CI : (0.6984, 0.7112)
    ##     No Information Rate : 0.2844          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6264          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8186   0.6397   0.6587   0.7410   0.6088
    ## Specificity            0.9109   0.9245   0.9012   0.9188   0.9742
    ## Pos Pred Value         0.7850   0.6703   0.5847   0.6415   0.8417
    ## Neg Pred Value         0.9267   0.9145   0.9259   0.9476   0.9171
    ## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2328   0.1238   0.1149   0.1214   0.1119
    ## Detection Prevalence   0.2966   0.1847   0.1965   0.1893   0.1330
    ## Balanced Accuracy      0.8648   0.7821   0.7799   0.8299   0.7915

Conclusion
----------

The random forest model had an excellent performance, with 99.98%
accuracy on the training dataset.

The linear discriminant analysis model had an inferior performance
compared to previous model, just 70% accuracy.

Based on the performance comparison above, the random forest model is
the best fitted model and it will be used to make the out of the sample
prediction, in this case, the testing dataset with new 20 samples.

Below are the random forest predictions for the outcome variable CLASSE,
using the testing dataset:

    # Based on the outstanding accuracy of the Random Forest model, I will use it to do the predictions on the testing dataset.
    predRftest <- predict(fitRf,test)
    predRftest

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
