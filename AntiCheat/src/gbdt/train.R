library(xgboost)
require(methods)
require(Matrix)
setwd("F:/RWorkspace/AntiCheat")

#Train Model
dtrain <- read.table("data/gbdt/Train.txt", header=FALSE, sep=",")
tlabel <- as.numeric(dtrain[[49]])
tdata <- as.matrix(dtrain[3:48])

txgmat <- xgb.DMatrix(tdata, label = tlabel)

param <- list("objective" = "binary:logistic",
              "bst:eta" = 0.05,
              "bst:max_depth" = 10,
              "eval_metric" = "auc",
              "eval_metric" = "error",
              "silent" = 1,
              "nthread" = 4)
watchlist <- list("train" = txgmat)

nround = 3000

bst = xgb.train(param, txgmat, nround, watchlist)

#save model
xgb.save(bst, 'model/gbdt/binarylogistic-005-10-3000.model')