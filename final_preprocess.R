library(data.table)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
fileName <- args[1]

print(paste("Reading", fileName))

Sam <- fread(paste0("data/splits/", fileName), header = T) # remove header

print("Done reading file.")
print(typeof(Sam))
print(class(Sam))
for (i in 1:100) {
  print(class(Sam[,i]))
}

unitScale <- function(v) {
  return((v - min(v)) / (max(v) - min(v)))
}

Sam <- Sam[, lapply(.SD, unitScale)]
# Sam <- apply(Sam, 2, unitScale)
write_csv(Sam, paste0("data/splits/P", fileName))



