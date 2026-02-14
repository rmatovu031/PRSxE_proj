
dir.create('/home/rstudio-server/genopred/config/', recursive = T)
dir.create('/home/rstudio-server/genopred/output', recursive = T)

# create target list
target_list <- data.frame(
  name='ukb',
  path='/home/rstudio-server/genopred/input/ukb_imp_afr',
  type='bgen',
  indiv_report=F,
  unrel= '/home/rstudio-server/ukb/ukb_symlinks/unrel_afr.txt'
)

write.table(
  target_list,
  '/home/rstudio-server/genopred/config/target_list.txt',
  col.names = T,
  row.names = F,
  quote = F
)

# http://www.diagram-consortium.org/downloads.html
# PUBMED ID: 38374256
# Title: Genetic drivers of heterogeneity in type 2 diabetes pathophysiology	
# main Author: Ken Suzuki

# Create gwas_list configuration with two entries
gwas_list <- data.frame(
  name = 'T2D_SUZUKI',
  path = '/home/rstudio-server/t2d_SUZUKI.gz',
  population = 'EUR',
  n = 2535601,
  sampling = 0.2,
  prevalence = 0.1,
  mean = NA,
  sd = NA,
  label = '"Multi"'
)

# Write to file
write.table(
  gwas_list,
  '/home/rstudio-server/genopred/config/gwas_list.txt',
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE,
  sep = ' '
)

config <- list(
  outdir = "/home/rstudio-server/genopred/output",
  config_file = "/home/rstudio-server/genopred/config/config.yaml",
  gwas_list = "/home/rstudio-server/genopred/config/gwas_list.txt",
  target_list = "/home/rstudio-server/genopred/config/target_list.txt",
  score_list = "/home/rstudio-server/genopred/config/score_list.txt",
  gwas_groups = "/home/rstudio-server/genopred/config/gwas_groups.txt",
  
  pgs_methods = c("ptclump", "dbslmm", "lassosum", "megaprs", "quickprs", "prscs", "bridgeprs"),
  ptclump_pts = c("5e-8", "1e-6", "1e-4", "1e-2", "0.1", "0.2", "0.3", "0.4", "0.5", "1"),
  dbslmm_h2f = c("0.8", "1", "1.2"),
  prscs_phi = c("1e-6", "1e-4", "1e-2", "1", "auto"),
  prscs_ldref = "1kg",
  
  ancestry_prob_thresh = 0.8,
  testing = NA,
  
  cores_prep_pgs = 10,
  cores_target_pgs = 10,
  mem_target_pgs = 10000,
  
  cores_outlier_detection = 5,
  
  pgs_scaling = "continuous",
  h2_method = "ldsc"
)

library(yaml)
write_yaml(config, "/home/rstudio-server/genopred/config/config.yaml")

