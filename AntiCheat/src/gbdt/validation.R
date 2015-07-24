library(xgboost)
require(gbm)
require(methods)
require(Matrix)
setwd("F:/RWorkspace/AntiCheat")

#load model
bst <- xgb.load('model/gbdt/binarylogistic-01-10-3000.model')

dvalidation <- read.table("data/gbdt/Validation.txt", header=FALSE, sep=",")
vlabel <- as.numeric(dvalidation[[49]])
vdata <- as.matrix(dvalidation[3:48])

vxgmat <- xgb.DMatrix(vdata, label = vlabel)

vpred <- predict(bst, vxgmat, missing = NULL, outputmargin = FALSE, ntreelimit = NULL, predleaf = FALSE)

str(vpred)
vpred[vpred>0.1] <- 1
vpred[vpred<=0.1] <- 0
vpred <- as.integer(vpred)
vreal <- vlabel

gbm.roc.area(vreal,vpred)

cmv <- data.frame(vpred, vreal)
ftable(cmv)