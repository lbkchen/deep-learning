library(data.table)

# For use with command line Rscript

args <- commandArgs(trailingOnly = TRUE)
path_sam <- args[1]
path_columns <- args[2]

Sam <- fread(path_sam)
Columns <- fread(path_columns)

Sam.names <- names(read.csv(path_sam, nrows = 1, fileEncoding = "UTF-8-BOM"))
Column.names <- names(read.csv(path_sam, nrows = 1, fileEncoding = "UTF-8-BOM"))

names(Sam) <- Sam.names
names(Columns) <- Column.names

print("Removed BOM from text")

Columns.vec <- Columns[,1]

Sam <- Sam[, Columns.vec, with=F]