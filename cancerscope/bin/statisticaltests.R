#####################
### GET PREDICTIONS and Tumor Content
#####################
source("/home/jgrewal/projects/bin/rscript_vis/ensemble_nn_tcga_withwithoutpog_functions.R", echo=FALSE)

tcga_pred <- get_predicted("/projects/jgrewal_prj/jgrewal_prj_results/expression_classifier/v1/paper/revisions/tcga_paperpreds_only/", look_recursive = FALSE)

#####################
### Accuracy and TC
#####################

#####################
### Accuracy and training class size
#####################

#####################
### Accuracy and confidnence score
#####################


