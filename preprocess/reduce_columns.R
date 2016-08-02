library(data.table)

# For use with command line Rscript

args <- commandArgs(trailingOnly = TRUE)
path_sam <- args[1]
path_columns <- args[2]
dest_filename <- args[3]

print("Reading files")
Sam <- fread(path_sam)
Columns <- fread(path_columns, header = FALSE)
print("Finished reading files")

Sam.names <- names(read.csv(path_sam, nrows = 1, fileEncoding = "UTF-8-BOM"))
Column.names <- c("features")

names(Sam) <- Sam.names
names(Columns) <- Column.names

print("Removed BOM from text")
print(str(Sam))
print(str(Columns))

Columns.vec <- Columns$features # first column
# print("Before")
# print(Columns.vec)
Columns.vec <- Columns.vec[which(Columns.vec %in% colnames(Sam))]
print("Reduced Columns vec")
# print("After")
# print(Columns.vec)

print("Filtering columns")
Sam <- Sam[, Columns.vec, with=F]
fwrite(Sam, file.path = dest_filename)
print("Done filtering columns")