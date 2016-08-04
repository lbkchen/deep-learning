library(data.table) # Must have data.table v1.9.7+
library(readr)
library(DMwR)
library(ROSE)

# Usage (must be run from command line)
# Rscript <path/to/sam_table.csv> <path/to/training_ids.csv> <path/to/testing_ids.csv> <optional: base name>
# Program will print steps of execution and write 5 different files to disk:
#   - Train x (saved as base_name_train_x.csv)
#   - Train y (saved as base_name_train_y.csv as one-hot vectors)
#   - Test x (saved as base_name_test_x.csv)
#   - Test y (saved as base_name_test_y.csv as one-hot vectors)
#   - Test ids (saved as base_name_test_ids.csv) (patient ids in order of all the test cases)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
path_sam <- args[1]
path_train_ids <- args[2]
path_test_ids <- args[3]
base_name <- args[4]

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
  if (is.factor(v)) {
    return(v)
  }
  range <- max(v) - min(v)
  if (range == 0) {
    return(0)
  }
  return((v - min(v)) / range)
}

print(str(Sam))

# Test min value of Sam
# Sam.maxs <- Sam[, lapply(.SD, max)]
# print(str(Sam.maxs))
# print(sum(Sam.maxs==0))

# Change y values of IP/ED to 1/0 depending on return or not (binarize)
Sam$ED_YTM <- ifelse(Sam$ED_YTM > 0, 1, 0)
Sam$IP_YTM <- ifelse(Sam$IP_YTM > 0, 1, 0)

# Change all necessary columns to factors to prevent scaling and 
# to assure SMOTE works
Sam$StatePatientID <- as.factor(Sam$StatePatientID)
Sam$ED_YTM <- as.factor(Sam$ED_YTM)
Sam$IP_YTM <- as.factor(Sam$IP_YTM)

# Scale all columns of Sam
print("Starting to scale table.")
Sam <- Sam[, lapply(.SD, unitScale)]
print("Completed scaling of columns.")

# Split into train and test
print("Starting to split into train and test sets.")
Sam.train <- Sam[StatePatientID %in% Train_ids[[1]]]
Sam.test <- Sam[StatePatientID %in% Test_ids[[1]]]
rm(Sam)
print("Finished splitting into train and test sets.")

# SMOTE algorithm for balancing training data by interpolated over/undersampling
# Smote parameters
# print("Beginning to apply SMOTE algorithm.")
# percent_to_oversample <- 500
# percent_ratio_major_to_minor <- 100
# Sam.train <- SMOTE(IP_YTM ~ . -StatePatientID -ED_YTM, data = Sam.train, 
#                    perc.over = percent_to_oversample, perc.under = percent_ratio_major_to_minor)
# print("Finished applying SMOTE algorithm.")

# ROSE algorithm for balancing training data by over/undersampling
print("Beginning to apply ROSE algorithm.")
result_sample_size <- 300000
rare_proportion <- 0.5
# Sam.train.without_factors <- Sam.train[, !c("StatePatientID", "ED_YTM"), with = FALSE]
# Sam.train.factors <- Sam.train[, c("StatePatientID", "ED_YTM"), with = FALSE]
Sam.train <- ovun.sample(ED_YTM ~ . -StatePatientID -IP_YTM, data = Sam.train, 
                         method = "both", N = result_sample_size, p = rare_proportion)$data
Sam.train <- data.table(Sam.train)
print("Finished applying ROSE algorithm.")

# Shuffle train data to homogenize 0/1 y values
print("Begin shuffle.")
Sam.train <- Sam.train[sample(nrow(Sam.train)),]
print("Finished shuffle.")

# Split into train.x, train.y, test.x, test.y
print("Begin split into x/y.")
Sam.train.x <- Sam.train[, !c("StatePatientID", "ED_YTM", "IP_YTM"), with = FALSE]
Sam.train.y <- Sam.train[, c("ED_YTM"), with = FALSE]
rm(Sam.train)
Sam.test.x <- Sam.test[, !c("StatePatientID", "ED_YTM", "IP_YTM"), with = FALSE]
Sam.test.y <- Sam.test[, c("ED_YTM"), with = FALSE]
Sam.test.ids <- Sam.test[, c("StatePatientID"), with = FALSE]
rm(Sam.test)
print("Finished split into x/y.")

# Change y to one-hot
Sam.train.y[, zero := ifelse(ED_YTM == 0, 1, 0)]
Sam.train.y[, one := ED_YTM]
Sam.train.y[, ED_YTM := NULL]
Sam.test.y[, zero := ifelse(ED_YTM == 0, 1, 0)]
Sam.test.y[, one := ED_YTM]
Sam.test.y[, ED_YTM := NULL]

# Write all splits to file
print("Begin write to file.")
base_name <- ifelse(is.na(base_name), "SAMFull", base_name)
fwrite(Sam.train.x, paste0(base_name, "_train_x", ".csv"), col.names = FALSE)
fwrite(Sam.train.y, paste0(base_name, "_train_y", ".csv"), col.names = FALSE)
fwrite(Sam.test.x, paste0(base_name, "_test_x", ".csv"), col.names = FALSE)
fwrite(Sam.test.y, paste0(base_name, "_test_y", ".csv"), col.names = FALSE)
fwrite(Sam.test.ids, paste0(base_name, "_test_ids", ".csv"))
print("Finished write to file.")

# Remove all columns with all zero entries 
# Sam <- Sam[,which(unlist(lapply(Sam, function(x)!all(is.zero(x))))),with=F]
# print(str(Sam))
