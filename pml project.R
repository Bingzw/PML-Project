#LOAD TRAINING AND TESTING DATA
setwd("C:/Users/Bingz/Documents/R_data/pratical ml")
train_raw <- read.csv("pml-training.csv")[, -1]
test_raw <- read.csv("pml-testing.csv")

#CLEANING TRAINING AND TESTING DATA
train_raw[train_raw == ""] <- NA
na_idx <- which(colSums(is.na(train_raw)) != 0)
identime_idx <- 1:6
train_clean <- train_raw[,-c(na_idx,identime_idx)]

test_raw[train_raw == ""] <- NA
na_idx_test <- which(colSums(is.na(test_raw)) != 0)
identime_idx_test <- 1:6
test_clean <- test_raw[,-c(na_idx_test,identime_idx_test)]

#FEATURE SELECTION
rfWithFilter <- sbf(train_clean, train_clean$classe, sbfControl = 
                      sbfControl(functions = rfSBF, method = "repeatedcv", repeats = 5))

#DISTRIBUTION OF ACCURACY OF 50 (10 FOLDS CV, REPEAT 5 TIMES) RESAMPLES
densityplot(rfWithFilter)

#OPTIMAL SELECTED VARIABLES
train_sel <- train_clean[rfWithFilter$optVariables]

#USE RANDOM FOREST MODEL WITH REPEATED CROSS VALIDATION
profile <- rfe(train_sel, train_clean$classe,
               sizes = c(10,20,30,40),
               rfeControl = rfeControl(functions=rfFuncs
                                       ,method="cv"))

#PLOT OF ACCURACY VESUS PREDICTOR SIZE
plot(profile, type = c("g", "o"))

#DENSITY OF ACCURACY AMONG 10 FOLDS
densityplot(profile)

#THE OPTIMAL PREDICTORS CHOSEN BY 10 FOLDS CV
profile$optVariables

#USE THE OPTIMAL MODEL TO PREDICT TEST SET
predict(profile$fit, test_clean)

