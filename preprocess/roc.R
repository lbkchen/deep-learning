library("ROCR")

args <- commandArgs(trailingOnly = TRUE)
pred_path <- args[1]
labels_path <- args[2]

pred <- read.csv(pred_path, header = FALSE)[,2]
labels <- read.csv(labels_path, header = FALSE)[,2]

pred <- prediction(pred, labels)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") # ROC
pdf("ROC.pdf")
plot(perf, col=rainbow(10))
dev.off()