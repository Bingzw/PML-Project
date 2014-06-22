




Human Activity Recognition -- Weight Lifting Exercise Learning
========================================================

Data Induction
---------------------------

The Weight Lifting Exercise Dataset was collected from the Human Activity Recognition Project in Groupware@LES. (http://groupware.les.inf.puc-rio.br/har) In this study, six young health participants, who were wearing accaccelerometers on the belt, forearm, arm, and dumbell, were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions. We want to investigate "how (well)" an activity was performed by the wearer. The manner they did the exercise (the "classe" variable in the dataset) is the outcome of interest. 159 features are given to help building the prediction algorithm.

Data Cleaning
----------------------

Before performing any analysis, we did some data cleaning in the first place. The following types of feature variables are excluded:
(1) Variables with missing values
(2) Identity and timestamp features

After the data cleaning step, 52 features are left for further analysis.

Model
------------------

We use the Random Forest learning algorithm to build the model. Generally the steps are as follows. We resample the training data, then rebuild classification or regression trees on each of the resampled data. When we split the data each time in a classification tree, we also resample the variables. A large number of trees are then built. We vote or average those trees in order to get the prediction for a new outcome. Random Forest method is highly accurate in many cases. However, it may counter several problems such as low speed and overfitting.

Feature Seclection
------------------

The method we apply in this section is to pre-screen the predictors using simple univariate statistical methods then only use those that pass some criterion in the subsequent model steps. The function applied in this step can be used to get resampling estimates for models when simple, filter-based feature selection is applied to the training data.

For each iteration of resampling, the predictor variables are univariately filtered prior to modeling. Performance of this approach is estimated using resampling. The same filter and model are then applied to the entire training set and the final model (and final features) are saved. In our analysis, we use repeated 10 folds cross validation with repeated times equals 5 to select features. 

After feature selection step, 46 variables are chosen in further analysis.



```r
rfWithFilter$optVariables
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_y"         
## [19] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [22] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [25] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [28] "total_accel_dumbbell" "gyros_dumbbell_y"     "accel_dumbbell_x"    
## [31] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [34] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [37] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [40] "gyros_forearm_x"      "accel_forearm_x"      "accel_forearm_y"     
## [43] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [46] "magnet_forearm_z"
```

```r
densityplot(rfWithFilter)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2.png) 

Cross Validation
---------------------

With regard to the potential overfitting problem of Random Forest, we perform a 10-fold cross validation to perform recursive feature selection. We use 10, 20, 30 and 40 as the cadidate subset sizes. The CV procedure is as follows:

1. Split the dataset into 10 dataset.  
2. Each time, 9/10 datasets are training data, 1/10 is testing. Do:  
   *Train the model on the training set using all predictors then predict the testing set  
   *Calculate the variable importance or rankings.  
   *For each subset size S(i), keep the S(i) most important variables, train the model on the training set using S(i) predictors, predict the testing set.  
3. Calculate the performance profile over the S(i) using test samples, determine the proper number of predictors.  
4. Estimate the final list of predictors to keep in the final model.  
5. Fit the final model based on the optimal S(i) using the original training set.

Here are the codes

```r
profile <- rfe(train_sel, train_clean$classe,
               sizes = c(10,20,30,40),
               rfeControl = rfeControl(functions=rfFuncs,method="cv"))
```
  
Result
---------------------------

The output and first figure below show that the best subset size was estimated to be 40 predictors. The density plot showed the prediction accuracy for these variables in 10 cross validation iterations. The prediction accuracy based on cross validation result on the 40 predictors selected is 0.997. The out-of-sample error is 0.003415.




```r
print(profile)
```

```

Recursive feature selection

Outer resampling method: Cross-Validated (10 fold) 

Resampling performance over subset size:

 Variables Accuracy Kappa AccuracySD KappaSD Selected
        10    0.993 0.991    0.00238 0.00301         
        20    0.995 0.993    0.00265 0.00335         
        30    0.996 0.995    0.00248 0.00313         
        40    0.997 0.996    0.00195 0.00247        *
        46    0.996 0.995    0.00176 0.00223         

The top 5 variables (out of 40):
   roll_belt, yaw_belt, magnet_dumbbell_z, pitch_belt, magnet_dumbbell_y
```

```r
(error <- (1 - subset(profile$results, Variables==40)$Accuracy))
```

```
[1] 0.003415
```

```r
plot(profile, type = c("g", "o"))
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-51.png) 

```r
densityplot(profile)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-52.png) 

The optimal 40 predictors are listed as follows:

```r
profile$optVariables
```

```
 [1] "roll_belt"            "yaw_belt"             "magnet_dumbbell_z"   
 [4] "pitch_belt"           "magnet_dumbbell_y"    "pitch_forearm"       
 [7] "accel_dumbbell_y"     "magnet_forearm_z"     "gyros_arm_y"         
[10] "roll_dumbbell"        "roll_forearm"         "roll_arm"            
[13] "accel_dumbbell_z"     "magnet_dumbbell_x"    "gyros_belt_z"        
[16] "magnet_belt_z"        "yaw_arm"              "magnet_belt_y"       
[19] "magnet_belt_x"        "accel_forearm_x"      "magnet_forearm_y"    
[22] "magnet_arm_z"         "yaw_dumbbell"         "gyros_dumbbell_y"    
[25] "accel_forearm_z"      "accel_belt_z"         "total_accel_dumbbell"
[28] "accel_forearm_y"      "yaw_forearm"          "total_accel_forearm" 
[31] "pitch_arm"            "accel_arm_y"          "accel_dumbbell_x"    
[34] "accel_arm_z"          "magnet_forearm_x"     "total_accel_arm"     
[37] "gyros_forearm_x"      "accel_arm_x"          "gyros_belt_x"        
[40] "magnet_arm_x"        
```

Prediction
---------------------------------

We will use the bulit algorithm to predict the 20 test samples. Here is the result.



```r
predict(profile$fit,test_clean)
```

```
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
Levels: A B C D E
```
