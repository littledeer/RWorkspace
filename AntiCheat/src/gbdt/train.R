library(xgboost)
require(methods)
require(Matrix)
setwd("F:/RWorkspace/AntiCheat")

#Train Model
dtrain <- read.table("data/gbdt/Train.txt", header=FALSE, sep=",")
#dtrain <- read.csv("data/gbdt/Train.csv", header=TRUE)
tlabel <- as.numeric(dtrain[[49]])
tdata <- as.matrix(dtrain[3:48])

txgmat <- xgb.DMatrix(tdata, label = tlabel)

param <- list("objective" = "binary:logistic",
              "bst:eta" = 0.1,
              "bst:max_depth" = 5,
              "eval_metric" = "auc",
              "eval_metric" = "error",
              "silent" = 1,
              "nthread" = 4)
watchlist <- list("train" = txgmat)

nround = 2000

bst = xgb.train(param, txgmat, nround, watchlist)

#save model
xgb.save(bst, 'model/gbdt/binarylogistic-01-5-2000.model')