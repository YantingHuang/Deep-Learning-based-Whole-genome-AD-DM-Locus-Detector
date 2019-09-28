library(data.table)

trait.names = c("amyloid", "braak", "cerad", "gpath", "tangles")
rda.data.dir <- "/home/ec2-user/volume/git/EnsembleCpG/data/features"
incomplete.train.data.dir <- "/home/ec2-user/yanting/AD_Deeplearning_regression/training_data"
rda.fns <- list.files(rda.data.dir, pattern="\\.rda$")
print(rda.fns)
for (trait in trait.names) {
  #trait <- "amyloid"
  print(trait)
  incomplete.train.data.fn <- file.path(incomplete.train.data.dir, sprintf("%s_training_data.csv", trait))
  train.data.dt <- fread(incomplete.train.data.fn)
  print(dim(train.data.dt))
  for (rda.fn in rda.fns) {
    full.rda.name <- file.path(rda.data.dir, rda.fn)
    print("loading data...")
    print(full.rda.name)
    load(full.rda.name) # load rda data
    print("finish loading...")
    if ("data.table" %in% class(readmat)) {
      train.data.dt<- cbind(train.data.dt, readmat[train.data.dt[["winid"]]])
    } else {
      print("data loaded as data.frame")
      train.data.dt<- cbind(train.data.dt, as.data.table(readmat[train.data.dt[["winid"]], ]))
    }
    print(dim(train.data.dt))
    # save RAM 
    rm(readmat)
    rm(allreads)
  }
  ### save to the disk
  print("saving to disk...")
  out.fn <- file.path(incomplete.train.data.dir, sprintf("%s_complete_training_data.csv", trait))
  fwrite(train.data.dt, file=out.fn, sep = ",")
  print("finished...")
  rm(train.data.dt)
}
