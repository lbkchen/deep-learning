library(readr)

args <- commandArgs(trailingOnly = TRUE)
fileName <- args[1]

Sam <- read_csv(paste0("data/splits/", fileName), skip = 1) # remove header

print("Done reading file.")

unitScale <- function(v) {
  return((v - min(v)) / (max(v) - min(v)))
}

Sam <- apply(Sam, 2, unitScale)
write_csv(Sam, paste0("data/splits/P", fileName))