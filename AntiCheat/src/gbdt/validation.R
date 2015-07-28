library(xgboost)
require(gbm)
require(methods)
require(Matrix)
setwd("F:/RWorkspace/AntiCheat")

#load model
bst <- xgb.load('model/gbdt/binarylogistic-01-5-2000.model')

dvalidation <- read.table("data/gbdt/Validation.txt", header=FALSE, sep=",")
vlabel <- as.numeric(dvalidation[[49]])
vdata <- as.matrix(dvalidation[3:48])

vxgmat <- xgb.DMatrix(vdata, label = vlabel)

vpred <- predict(bst, vxgmat, missing = NULL, outputmargin = FALSE, ntreelimit = NULL, predleaf = FALSE)

vpred[vpred>0.01] <- 1
vpred[vpred<=0.01] <- 0
vpred <- as.integer(vpred)
vreal <- vlabel

print(gbm.roc.area(vreal,vpred))

cmv <- data.frame(vpred, vreal)
print(ftable(cmv))