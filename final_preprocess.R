library(data.table)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
fileName <- args[1]

print(paste("Reading", fileName))

first.line <- fread(paste0("data/splits/", fileName), header = T, nrows = 2)
print(ncol(first.line))
Sam <- fread(paste0("data/splits/", fileName), header = T, colClasses = rep("numeric", ncol(first.line))) # remove header

print("Done reading file.")
print(typeof(Sam))
print(class(Sam))
for (i in 1:100) {
  print(class(Sam[,i]))
}

unitScale <- function(v) {
  range <- max(v) - min(v)
  if (range == 0) {
    return(0)
  }
  return((v - min(v)) / range)
}

print("Starting to scale table.")
Sam <- Sam[, lapply(.SD, unitScale)]
# Sam <- apply(Sam, 2, unitScale)
write_csv(Sam, paste0("data/splits/P", fileName))
print("Written to file.")


