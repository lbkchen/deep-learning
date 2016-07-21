library(data.table)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
path_sam <- args[1]
path_train_ids <- args[2]
path_test_ids <- args[3]

print(paste("Reading", path_sam))
Sam <- fread(path_sam, header = T)
print(paste("Reading", path_train_ids))
Train_ids <- fread(path_train_ids, header = T)
print(paste("Reading", path_test_ids))
Test_ids <- fread(path_test_ids, header = T)

print("Done reading files.")

is.zero <- function(v) {
  return(v==0)
}

unitScale <- function(v) {
  range <- max(v) - min(v)
  if (range == 0) {
    return(0)
  }
  return((v - min(v)) / range)
}

print(str(Sam))

# Test min value of Sam
Sam.maxs <- Sam[, lapply(.SD, max)]
print(str(Sam.maxs))
print(sum(Sam.maxs==0))

# Scale all columns of sam
print("Starting to scale table.")
Sam <- Sam[, lapply(.SD, unitScale)]

# Split into train and test
print("Staring to split into train and test sets.")
Sam.train <- Sam[StatePatientID %in% Train_ids,,with=T]
Sam.test <- Sam[StatePatientID %in% Test_ids,, with=T]
rm(Sam)

# Remove all columns with all zero entries 
# Sam <- Sam[,which(unlist(lapply(Sam, function(x)!all(is.zero(x))))),with=F]
# print(str(Sam))
