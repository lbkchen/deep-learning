library(data.table)
library(dplyr)

Sam <- fread("data/ssSAMTablePart01.csv")

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

# Returns whether a vector should be scaled in the SAM
shouldScale <- function(v) {
  
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

isNotDenseOrBinary <- function(v) {
  return(notOnlyBinary(v) && !isDense90(v))
}

# Performs pre-processing steps on vector based on data format
preprocess <- function(v) {
  if (isDense90(v)) {
    v <- standardize(v)
  }
  if (isNotDenseOrBinary(v)) {
    v <- unitScale(v)
  }
  return(v)
}

scaleAndNormalize <- function(dtf) {
  return(dtf %>% vapply(preprocess, numeric(nrow(dtf))))
}

###################
# PREPROCESS DATA #
###################

# Only keep numeric data
Sam <- Sam %>% select_if(hasAllNumeric)

# Split data into labels and features
Sam.ys <- Sam[, 1:3, with=F]
Sam.xs <- Sam[, 4:ncol(Sam), with=F]

# preprocess <- function(dtf) {
#   dtf <- dtf %>%
#     mutate_if(isDense90, standardize) %>%
#     mutate_if(function(v){notOnlyBinary(v) & !isDense90(v)}, unitScale)
#     return(dtf)
# }

# Sam.xs %>%
#   mutate_if(isDense90, standardize) %>%
#   mutate_if(isNotDenseOrBinary, unitScale) %>%
#   write.csv("data/SAMX.csv")

Sam.xs %>%
  scaleAndNormalize() %>%
  write.csv("data/SAMX.csv")

# write.csv(mutate_if(mutate_if(Sam.xs, isDense90, standardize), isNotDenseOrBinary, unitScale), "data/SAMX.csv")
write.csv(Sam.ys, "data/SAMY.csv")
