library(rlist)
library(mosaic)
library(dplyr)

Sam <- read.file("data/sSAMTablePart01.csv")

####################
# HELPER FUNCTIONS #
####################

# Summarizes a vector
summCol <- function(v) {
  print(paste("Mean:", mean(v)))
  print(paste("Standard deviation:", sd(v)))
}

# Returns whether a vector has no NA values
hasAllNumeric <- function(v) {
  return(!anyNA(v) & is.numeric(v))
}

# Takes a boolean return function f and returns the number of columns in df
# with f true
numColsWith <- function(df, f) {
  return(Sam %>% sapply(f) %>% sum())
}

onlyBinary <- function(v) {
  return(sum(v == 0) + sum(v == 1) == length(v))
}

notOnlyBinary <- function(v) {
  return(!onlyBinary(v))
}

# Scales a vector to standard z units
standardize <- function(v) {
  return((v - mean(v)) / sd(v))
}

# Scales a vector to entries between 0 and 1
unitScale <- function(v) {
  return((v - min(v)) / (max(v) - min(v)))
}

# Returns whether a vector has >= threshold proportion of entries 0
isSparse <- function(v, threshold) {
  propZero <- length(v[v==0]) / length(v)
  return(propZero >= threshold)
}

isSparse90 <- function(v) {
  return(isSparse(v, 0.90))
}

# Returns whether a vector has > 0.90 proportion of entries nonzero
isDense90 <- function(v) {
  return(!isSparse(v, 0.10))
}

##############################
# GET INFORMATION ABOUT DATA #
##############################

Sam <- Sam %>% select_if(hasAllNumeric)

print(paste("Num cols with only numerical entries:", numColsWith(Sam, function(v){
  is.numeric(v)
})))

print(paste("Num cols with only zero entries:", numColsWith(Sam, function(v){
  mean(v) == 0
})))

print(paste("Num cols with negative entries:", numColsWith(Sam, function(v){
  any(v < 0)
})))

print(paste("Num cols with binary values:", numColsWith(Sam, onlyBinary)))
print(paste("Num cols with non-binary values:", numColsWith(Sam, notOnlyBinary)))

SamNonBinary <- Sam %>% select_if(notOnlyBinary)

SummaryOfNonBinary <- data.frame(
  name=names(SamNonBinary), 
  mean=SamNonBinary %>% sapply(mean), 
  sd=SamNonBinary %>% sapply(sd), 
  min=SamNonBinary %>% sapply(min), 
  q1=SamNonBinary %>% sapply(function(v){quantile(v)["25%"]}), 
  median=SamNonBinary %>% sapply(median),
  q3=SamNonBinary %>% sapply(function(v){quantile(v)["75%"]}), 
  max=SamNonBinary %>% sapply(max)
)

write.csv(SummaryOfNonBinary, "data/summary_of_non_binary.csv")

###################
# PREPROCESS DATA #
###################

preprocess <- function(df) {
  df <- df %>%
    mutate_if(isDense90, standardize) %>%
    mutate_if(function(v){notOnlyBinary(v) & !isDense90(v)}, unitScale)
}

Sam <- preprocess(Sam)
