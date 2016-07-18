library(data.table)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
fileName <- args[1]
destName <- args[2]

dtf <- fread(fileName, header=T)

dtf <- dtf[, zero := ifelse(IP_YTM == 0, 1, 0)]
dtf <- dtf[, one := ifelse(IP_YTM == 1, 1, 0)]
dtf <- dtf[,  IP_YTM := NULL]

write_csv(dtf, destName)
print("Written to file.")