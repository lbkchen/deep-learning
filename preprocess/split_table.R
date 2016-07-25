library(data.table)
library(readr)
source("")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
path_sam <- args[1]
path_train_ids <- args[2]
path_test_ids <- args[3]

# Read in raw files: SAM table, train case ids, and test case ids
print(paste("Reading", path_sam))
Sam <- fread(path_sam, header = T)
print(paste("Reading", path_train_ids))
Train_ids <- fread(path_train_ids, header = T)
print(paste("Reading", path_test_ids))
Test_ids <- fread(path_test_ids, header = T)

print("Done reading files.")

# Reset headers of data tables to get rid of BOM in case it's there
# http://stackoverflow.com/questions/21624796/read-the-text-file-with-bom-in-r
Sam.names <- names(read.csv(path_sam, nrows = 1, fileEncoding = "UTF-8-BOM"))
Train_ids.names <- names(read.csv(path_train_ids, nrows = 1, fileEncoding = "UTF-8-BOM"))
Test_ids.names <- names(read.csv(path_test_ids, nrows = 1, fileEncoding = "UTF-8-BOM"))

names(Sam) <- Sam.names
names(Train_ids) <- Train_ids.names
names(Test_ids) <- Test_ids.names

print("Removed BOM from text")

# Pre-processing functions
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

# Split into train and test
print("Staring to split into train and test sets.")
Sam.train <- Sam[StatePatientID %in% Train_ids,,with=T]
Sam.test <- Sam[StatePatientID %in% Test_ids,, with=T]
rm(Sam)

str(Sam.train)
str(Sam.test)

# Remove all columns with all zero entries 
# Sam <- Sam[,which(unlist(lapply(Sam, function(x)!all(is.zero(x))))),with=F]
# print(str(Sam))
