library(xgboost)
require(gbm)
require(methods)
require(Matrix)
setwd("F:/RWorkspace/AntiCheat")

#load model
bst <- xgb.load('model/gbdt/binarylogistic-005-10-1000.model')

dvalidation <- read.table("data/gbdt/Validation.txt", header=FALSE, sep=",")
vlabel <- as.numeric(dvalidation[[49]])
vdata <- as.matrix(dvalidation[3:48])

vxgmat <- xgb.DMatrix(vdata, label = vlabel)

vpred <- predict(bst, vxgmat, missing = NULL, outputmargin = FALSE, ntreelimit = NULL, predleaf = FALSE)

vpred[vpred>0.005] <- 1
vpred[vpred<=0.005] <- 0
vpred <- as.integer(vpred)
vreal <- vlabel

print(gbm.roc.area(vreal,vpred))

cmv <- data.frame(vpred, vreal)
print(ftable(cmv))