library(data.table)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
path_sam <- args[1]
path_train_ids <- args[2]
path_test_ids <- args[3]

print(paste("Reading", path_sam))
sam <- fread(path_sam, header = T)
print(paste("Reading", path_train_ids))
train_ids <- fread(path_train_ids, header = T)
print(paste("Reading", path_test_ids))
test_ids <- fread(path_test_ids, header = T)

print("Done reading files.")

is.zero <- function(v) {
  return(v==0)
}

print(str(sam))
# Remove all columns with all zero entries 
sam <- sam[,which(unlist(lapply(sam, function(x)!all(is.zero(x))))),with=F]
print(str(sam))
