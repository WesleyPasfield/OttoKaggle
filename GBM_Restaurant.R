## Load in data

setwd("~/Documents/Kaggle/")
data <- read.csv("Restauraunt_Train.csv")
testData <- read.csv("test.csv")

summary(data)
summary(testData)
ncol(testData)

## Install necessary packages

## Convert to Dates
## 3/28 Change dates to time since open rather than year

data$year <- as.Date(data$Open.Date, "%m/%d/%y")
testData$year <- as.Date(testData$Open.Date, "%m/%d/%y")

head(data)
## Remove unnecessary columns

trainNew <- data[,4:44]
testDataNew <- testData[,4:43]

## Convert Year to correct format
library(lubridate)
trainNew$today <- Sys.Date()
testDataNew$today <- Sys.Date()
trainNew$daysOpen <- trainNew$today - trainNew$year
testDataNew$daysOpen <- testDataNew$today - testDataNew$year
trainNew$daysOpen <- as.numeric(trainNew$daysOpen)
head(trainNew)
head(testDataNew)

trainNumeric <- trainNew[,c(1:39, 43)] 
testNumeric <- testDataNew[,c(1:39, 42)]

head(trainNumeric)
head(testNumeric)
revenue <- trainNew[,40]

## Convert data to numeric format for both training & test sets

trainNumbers <- sapply(trainNumeric, as.numeric)
trainModel <- cbind(trainNumbers, revenue)
trainModel <- as.data.frame(trainModel)
testNumeric <- sapply(testNumeric, as.numeric)
testNumeric <- as.data.frame(testNumeric)
summary(trainModel)
summary(testNumeric)
head(trainModel)
discrete <- trainModel[,40]
discrete <- as.data.frame(discrete)
## Run GBM model

## For 3/26 - Standardize all inputs and rerun
## For future - PCA
library(caret)
BC <- preProcess(trainModel[],
           method = c("center", "scale"),
           na.remove = TRUE,
           )

TC <- preProcess(testNumeric,
                 method = c("center", "scale"),
                 na.remove = TRUE,
)
BC <- as.numeric(BC)
trainer <- cbind(trainModel,BC)
PC <- predict(BC, trainModel[,-41])
TC <- predict(TC, testNumeric)
head(PC)

trainPCA <- cbind(PC,trainModel[,41])
names(trainPCA)[41] <- "revenue"
summary(trainPCA)

library(gbm)
library(dismo)
library(caret)

newGBM <- expand.grid(.n.trees = c(100,150,200,250,300),
                      .interaction.depth = 3:6,
                      .shrinkage = c(.001,.003,.005,.007,.009))

trainedGBM <- train(revenue~., method = "gbm", distribution = "gaussian",
                    data = trainModel, metric = "RMSE", tuneGrid = newGBM,
                    trControl = trainControl(method = "repeatedcv", number = 5,
                                             repeats = 2, verboseIter = FALSE,
                                             returnResamp = "all"))
predictGBM <- gbm(revenue~., distribution = "gaussian", data=trainModel, n.trees = 150,
                  interaction.depth = 6, shrinkage = .007)

head(trainModel)
## Evaluate Model
print(trainedGBM)
summary(predictGBM)


## Create Predictions

predictions <- predict(predictGBM, newdata = testNumeric, n.trees = 150)
submission <- as.data.frame(cbind(testData[,1], predictions))
colnames(submission) <- c("Id", "Prediction")
write.csv(submission, "submissionEight.csv", row.names = FALSE, quote = FALSE)

check <- read.csv("submissionEight.csv")
head(check)

## Discretized Attempt

library(discretization)

disc <- disc.Topdown(trainModel, method = 1)
discModel <- as.data.frame(disc$Disc.data)
discModel <- cbind(discModel[,1:40], trainModel[,41])

names(discModel)[names(discModel) == "trainModel[, 41]"] <- "revenue"
head(trainModel)
head(discModel)

newGBM1 <- gbm(revenue~., distribution = "gaussian", data = discTwo, var.monotone = NULL,
              n.trees = 300, interaction.depth = 4, shrinkage = .005, cv.folds = 5)

## Evaluate Model

summary.gbm(newGBM1)

## Create Prediction & submission

predictions <- predict(newGBM1, newdata = testNumeric)
submission <- as.data.frame(cbind(testData[,1], predictions))
colnames(submission) <- c("Id", "Prediction")
write.csv(submission, "submissionDiscModel.csv", row.names = FALSE, quote = FALSE)

check <- read.csv("submissionDiscModel.csv")
head(check)

## Just Discretize Year

discTwo <- cbind(trainModel, discModel[40])
discTwo <- discTwo[,-40]


