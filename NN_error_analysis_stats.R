##NN error analysis
rm(list=ls())
setwd("/Users/joemorford/Desktop/Nav_NNs/2023")

library(circular)
library(cowplot)
library(plyr)
library(ggplot2)
source("NN_error_analysis_stats_functions.R")


###

Test_data <- read.csv("Testing_data.csv")
head(Test_data) 
#Locations in X and Y of training points plus angs and dists to target and values of simulated gradients
All_mod_preds <- read.csv("All_model_predictions.csv")
head(All_mod_preds)
#Orientation predictions of different nav models at test points
#"TRUE." is the perfectly accurate solution

B1_mods <- which(ldply(strsplit(colnames(All_mod_preds), "_"), rbind)[,2]=="B1")
B2i_mods <- which(ldply(strsplit(colnames(All_mod_preds), "_"), rbind)[,2]=="B2i")
B2ii_mods <- which(ldply(strsplit(colnames(All_mod_preds), "_"), rbind)[,2]=="B2ii")
B3i_mods <- which(ldply(strsplit(colnames(All_mod_preds), "_"), rbind)[,2]=="B3i")
B3ii_mods <- which(ldply(strsplit(colnames(All_mod_preds), "_"), rbind)[,2]=="B3ii")
B4i_mods <- which(ldply(strsplit(colnames(All_mod_preds), "_"), rbind)[,2]=="B4i")
B4ii_mods <- which(ldply(strsplit(colnames(All_mod_preds), "_"), rbind)[,2]=="B4ii")
#Indices of different models

###

filenames_cont <- list.files("NN_continuous_2023_out", pattern="*.csv", full.names=TRUE)
import_func_out_cont <- import_process_func(filenames_cont)
#Importing neural network output files (continuous implementation)
meta_nets_cont <- import_func_out_cont[[2]]
func_out_cont <- abs_error_function(import_func_out_cont[[1]], meta_nets_cont, All_mod_preds, B1_mods, B2i_mods, B2ii_mods, 
                              B3i_mods, B3ii_mods, B4i_mods, B4ii_mods)
all_out_cont <- func_out_cont[[1]]
nn_model_comparisons_absdiff_means_cont <- func_out_cont[[2]]
#Calculates network performance and stuff

filenames_categ <- list.files("NN_categ_2023_out", pattern="*.csv", full.names=TRUE)
import_func_out_categ <- import_process_func(filenames_categ)
#Importing neural network output files (categoric implementation)
meta_nets_categ <- import_func_out_categ[[2]]
func_out_categ <- abs_error_function(import_func_out_categ[[1]], meta_nets_categ, All_mod_preds, B1_mods, B2i_mods, B2ii_mods, 
                                    B3i_mods, B3ii_mods, B4i_mods, B4ii_mods)
all_out_categ <- func_out_categ[[1]]
nn_model_comparisons_absdiff_means_categ <- func_out_categ[[2]]
#Calculates network performance and stuff

############




###### B1

B1s_cont <- which(meta_nets_cont$grad2=="B1")
B1_means_cont <- nn_model_comparisons_absdiff_means_cont[nn_model_comparisons_absdiff_means_cont$grad2=="B1",]

B1s_categ <- which(meta_nets_categ$grad2=="B1")
B1_means_categ <- nn_model_comparisons_absdiff_means_categ[nn_model_comparisons_absdiff_means_categ$grad2=="B1",]

B1_cont_lmms <- lmm_models_func("B1", meta_nets_cont, B1_means_cont, all_out_cont)
B1_cont_lmms
B1_categ_lmms <- lmm_models_func("B1", meta_nets_categ, B1_means_categ, all_out_categ)
B1_categ_lmms
#Stats models of model predictions of neural net outputs
#Formula: Mean absolute diff (across test locations for each neural net) ~ Model + (1 | Neural net ID)
#Mean absolute diff is between model prediction and neural net output across test locations for each neural net
#Whichever is smallest is best fitting model
#"TRUE." is perfectly accurate solution - input as nav model - this is used as a baseline to assess performance of other models


B1_cont_plot <- mod_error_plots(B1_means_cont, c(B1_means_cont$TRUE., B1_means_cont$A_B1_DIREC),
                list("TRUE", "DIREC"), list("NA", "NA"), 'none', 'none')
B1_cont_plot
#Learning curve

B1_cont_plot2 <- mod_error_plots2(B1_means_cont, c(B1_means_cont$A_B1_DIREC),
                                list("DIREC"), list("NA"), 'legend', 'none')
B1_cont_plot2
#Performance against baseline of perfectly accurate solution

B1_cont_plot3 <- mod_error_plots3(B1_means_cont, c(B1_means_cont$A_B1_DIREC),
                                          list("DIREC"), list("NA"), 'legend', 'none')
B1_cont_plot3

B1_categ_plot <- mod_error_plots(B1_means_categ, c(B1_means_categ$TRUE., B1_means_categ$A_B1_DIREC),
                                    list("TRUE", "DIREC"), list("NA", "NA"), 'legend', 'none')
B1_categ_plot
B1_categ_plot2 <- mod_error_plots2(B1_means_categ, c(B1_means_categ$A_B1_DIREC),
                                 list("DIREC"), list("NA"), 'legend', 'none')
B1_categ_plot2
B1_categ_plot3 <- mod_error_plots3(B1_means_categ, c(B1_means_categ$A_B1_DIREC),
                                               list("DIREC"), list("NA"), 'legend', 'none')
B1_categ_plot3


#png(filename="Write up/B1_relative_results.png", width=6.5, height=3,units="in",res=600)
plot_grid(B1_cont_plot2, B1_categ_plot2, nrow=1,
          labels = c('B1 with Continuous Output', 'B1 with Categoric Output'), vjust=2, hjust=-0.2)
#dev.off()

png(filename="Write up/B1_April24_results.png", width=6.5, height=3,units="in",res=600)
plot_grid(B1_cont_plot3, B1_categ_plot3, nrow=1,
          labels = c('B1 under Implementation 1', 'B1 under Implementation 2'),
          vjust=2, hjust=-0.2)
dev.off()

###### B2i

B2is_cont <- which(meta_nets_cont$grad2=="B2i")
B2i_means_cont <- nn_model_comparisons_absdiff_means_cont[nn_model_comparisons_absdiff_means_cont$grad2=="B2i",]

B2is_categ <- which(meta_nets_categ$grad2=="B2i")
B2i_means_categ <- nn_model_comparisons_absdiff_means_categ[nn_model_comparisons_absdiff_means_categ$grad2=="B2i",]

B2i_cont_lmms <- lmm_models_func("B2i", meta_nets_cont, B2i_means_cont, all_out_cont)
B2i_cont_lmms
B2i_categ_lmms <- lmm_models_func("B2i", meta_nets_categ, B2i_means_categ, all_out_categ)
B2i_categ_lmms

B2i_cont_sig_vs_T <- c(0, 3, 3, 3, 3, 1, 0, 0, 0) #from lmms - significance (number of stars) vs baseline at each training stage
B2i_categ_sig_vs_T <- c(0, 3, 0, 0, 0, 0, 0, 0, 0) #from lmms - significance (number of stars) vs baseline at each training stage

B2i_cont_sig_best_mod_comb <- c("APPROX", "APPROX", "APPROX", "APPROX", "APPROX") #from lmms - best performing model at each training stage
B2i_cont_sig_best_mod_extrap <- c("NA", "NA", "NA", "NA", "NA") #from lmms - best performing model at each training stage
B2i_categ_sig_best_mod_comb <- c("APPROX") #from lmms - best performing model at each training stage
B2i_categ_sig_best_mod_extrap <- c("NA") #from lmms - best performing model at each training stage

B2i_cont_sig_vs_All <- c(0, 2, 3, 3, 3, 3, 0, 0, 0) #from lmms - significance (number of stars) vs next best model at each training stage
B2i_categ_sig_vs_All <- c(0, 2, 0, 0, 0, 0, 0, 0, 0) #from lmms - significance (number of stars) vs next best model at each training stage

B2i_cont_plot <- mod_error_plots(B2i_means_cont, c(B2i_means_cont$TRUE., B2i_means_cont$A_B2i_DIREC, B2i_means_cont$A_B2i_APPROX),
                                list("TRUE", "DIREC", "APPROX"), list("NA", "NA", "NA"), 'none', 'none', B2i_cont_sig_vs_T, B2i_cont_sig_vs_All,
                                T, B2i_cont_sig_best_mod_extrap, B2i_cont_sig_best_mod_comb)
B2i_cont_plot
B2i_cont_plot2 <- mod_error_plots2(B2i_means_cont, c(B2i_means_cont$A_B2i_DIREC, B2i_means_cont$A_B2i_APPROX),
                                 list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2i_cont_sig_vs_T, B2i_cont_sig_vs_All,
                                 T, B2i_cont_sig_best_mod_extrap, B2i_cont_sig_best_mod_comb)
B2i_cont_plot2


B2i_cont_plot3 <- mod_error_plots3(B2i_means_cont, c(B2i_means_cont$A_B2i_DIREC, B2i_means_cont$A_B2i_APPROX),
                                           list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2i_cont_sig_vs_T, B2i_cont_sig_vs_All)
B2i_cont_plot3

B2i_categ_plot <- mod_error_plots(B2i_means_categ, c(B2i_means_categ$TRUE., B2i_means_categ$A_B2i_DIREC, B2i_means_categ$A_B2i_APPROX),
                                 list("TRUE", "DIREC", "APPROX"), list("NA", "NA", "NA"), 'legend', 'none', B2i_categ_sig_vs_T, B2i_categ_sig_vs_All,
                                 T, B2i_categ_sig_best_mod_extrap, B2i_categ_sig_best_mod_comb)
B2i_categ_plot
B2i_categ_plot2 <- mod_error_plots2(B2i_means_categ, c(B2i_means_categ$A_B2i_DIREC, B2i_means_categ$A_B2i_APPROX),
                                  list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2i_categ_sig_vs_T, B2i_categ_sig_vs_All,
                                  T, B2i_categ_sig_best_mod_extrap, B2i_categ_sig_best_mod_comb)
B2i_categ_plot2
B2i_categ_plot3 <- mod_error_plots3(B2i_means_categ, c(B2i_means_categ$A_B2i_DIREC, B2i_means_categ$A_B2i_APPROX),
                                   list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2i_categ_sig_vs_T, B2i_categ_sig_vs_All)
B2i_categ_plot3

#Orientation error plots of single NNs and population at specific stages of training where approximate model is best performing


B2i_plot_approx_cont_1000_nn1 <- directional_error_plots("B2i", 1000, All_mod_preds$A_B2i_APPROX, All_mod_preds$A_B2i_DIREC, 
                                                         meta_nets_cont, All_mod_preds, import_func_out_cont[[1]], 1, 'blue',
                                                         rowMeanCirc)
B2i_plot_approx_cont_1000_nn1
B2i_plot_approx_cont_1000_nn2 <- directional_error_plots("B2i", 1000, All_mod_preds$A_B2i_APPROX, All_mod_preds$A_B2i_DIREC, 
                                                         meta_nets_cont, All_mod_preds, import_func_out_cont[[1]], 2, 'blue',
                                                         rowMeanCirc)
B2i_plot_approx_cont_1000_nn2
B2i_plot_approx_cont_1000_pop <- directional_error_plots("B2i", 1000, All_mod_preds$A_B2i_APPROX, All_mod_preds$A_B2i_DIREC, 
                                                    meta_nets_cont, All_mod_preds, import_func_out_cont[[1]], 1:50, 'black', 
                                                    rowMeanCirc)
B2i_plot_approx_cont_1000_pop

B2i_plot_approx_categ_100_nn1 <- directional_error_plots("B2i", 100, All_mod_preds$A_B2i_APPROX, All_mod_preds$A_B2i_DIREC, 
                                                         meta_nets_categ, All_mod_preds, import_func_out_categ[[1]], 1, 'blue',
                                                         rowMeanCirc)
B2i_plot_approx_categ_100_nn1
B2i_plot_approx_categ_100_nn2 <- directional_error_plots("B2i", 100, All_mod_preds$A_B2i_APPROX, All_mod_preds$A_B2i_DIREC, 
                                                         meta_nets_categ, All_mod_preds, import_func_out_categ[[1]], 2, 'blue',
                                                         rowMeanCirc)
B2i_plot_approx_categ_100_nn2
B2i_plot_approx_categ_100_pop <- directional_error_plots("B2i", 100, All_mod_preds$A_B2i_APPROX, All_mod_preds$A_B2i_DIREC, 
                                                 meta_nets_categ, All_mod_preds, import_func_out_categ[[1]], 1:50, 'black', 
                                                 rowMeanCirc)
B2i_plot_approx_categ_100_pop

###### B2ii

B2iis_cont <- which(meta_nets_cont$grad2=="B2ii")
B2ii_means_cont <- nn_model_comparisons_absdiff_means_cont[nn_model_comparisons_absdiff_means_cont$grad2=="B2ii",]

B2iis_categ <- which(meta_nets_categ$grad2=="B2ii")
B2ii_means_categ <- nn_model_comparisons_absdiff_means_categ[nn_model_comparisons_absdiff_means_categ$grad2=="B2ii",]

B2ii_cont_lmms <- lmm_models_func("B2ii", meta_nets_cont, B2ii_means_cont, all_out_cont)
B2ii_cont_lmms
B2ii_categ_lmms <- lmm_models_func("B2ii", meta_nets_categ, B2ii_means_categ, all_out_categ)
B2ii_categ_lmms

B2ii_cont_sig_vs_T <- c(0, 3, 3, 3, 3, 3, 0, 0, 0)
B2ii_categ_sig_vs_T <- c(0, 3, 3, 0, 0, 0, 0, 0, 0)

B2ii_cont_sig_best_mod_comb <- c("APPROX", "APPROX", "APPROX", "APPROX", "APPROX")
B2ii_cont_sig_best_mod_extrap <- c("NA", "NA", "NA", "NA", "NA")
B2ii_categ_sig_best_mod_comb <- c("APPROX", "APPROX")
B2ii_categ_sig_best_mod_extrap <- c("NA", "NA")

B2ii_cont_sig_vs_All <- c(0, 0, 0, 0, 2, 3, 0, 0, 0)
B2ii_categ_sig_vs_All <- c(0, 0, 0, 0, 0, 0, 0, 0, 0)


B2ii_cont_plot <- mod_error_plots(B2ii_means_cont, c(B2ii_means_cont$TRUE., B2ii_means_cont$A_B2ii_DIREC, B2ii_means_cont$A_B2ii_APPROX),
                                 list("TRUE", "DIREC", "APPROX"), list("NA", "NA", "NA"), 'none', 'none', B2ii_cont_sig_vs_T, B2ii_cont_sig_vs_All,
                                 T, B2ii_cont_sig_best_mod_extrap, B2ii_cont_sig_best_mod_comb)
B2ii_cont_plot
B2ii_cont_plot2 <- mod_error_plots2(B2ii_means_cont, c(B2ii_means_cont$A_B2ii_DIREC, B2ii_means_cont$A_B2ii_APPROX),
                                  list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2ii_cont_sig_vs_T, B2ii_cont_sig_vs_All,
                                  T, B2ii_cont_sig_best_mod_extrap, B2ii_cont_sig_best_mod_comb)
B2ii_cont_plot2

B2ii_cont_plot3 <- mod_error_plots3(B2ii_means_cont, c(B2ii_means_cont$A_B2ii_DIREC, B2ii_means_cont$A_B2ii_APPROX),
                                   list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2ii_cont_sig_vs_T, B2ii_cont_sig_vs_All)
B2ii_cont_plot3

B2ii_categ_plot <- mod_error_plots(B2ii_means_categ, c(B2ii_means_categ$TRUE., B2ii_means_categ$A_B2ii_DIREC, B2ii_means_categ$A_B2ii_APPROX),
                                  list("TRUE", "DIREC", "APPROX"), list("NA", "NA", "NA"), 'none', 'none', B2ii_categ_sig_vs_T, B2ii_categ_sig_vs_All,
                                  T, B2ii_categ_sig_best_mod_extrap, B2ii_categ_sig_best_mod_comb)
B2ii_categ_plot
B2ii_categ_plot2 <- mod_error_plots2(B2ii_means_categ, c(B2ii_means_categ$A_B2ii_DIREC, B2ii_means_categ$A_B2ii_APPROX),
                                   list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2ii_categ_sig_vs_T, B2ii_categ_sig_vs_All,
                                   T, B2ii_categ_sig_best_mod_extrap, B2ii_categ_sig_best_mod_comb)
B2ii_categ_plot2
B2ii_categ_plot3 <- mod_error_plots3(B2ii_means_categ, c(B2ii_means_categ$A_B2ii_DIREC, B2ii_means_categ$A_B2ii_APPROX),
                                    list("DIREC", "APPROX"), list("NA", "NA"), 'legend', 'none', B2ii_categ_sig_vs_T, B2ii_categ_sig_vs_All)
B2ii_categ_plot3


B2ii_plot_approx_cont_2000_nn1 <- directional_error_plots("B2ii", 2000, All_mod_preds$A_B2ii_APPROX, All_mod_preds$A_B2ii_DIREC,
                                                          meta_nets_cont, All_mod_preds, import_func_out_cont[[1]], 1, 'blue', 
                                                          rowMeanCirc)
B2ii_plot_approx_cont_2000_nn1

B2ii_plot_approx_cont_2000_nn2 <- directional_error_plots("B2ii", 2000, All_mod_preds$A_B2ii_APPROX, All_mod_preds$A_B2ii_DIREC,
                                                          meta_nets_cont, All_mod_preds, import_func_out_cont[[1]], 2, 'blue',
                                                          rowMeanCirc)
B2ii_plot_approx_cont_2000_nn2


B2ii_plot_approx_cont_2000_pop <- directional_error_plots("B2ii", 2000, All_mod_preds$A_B2ii_APPROX, All_mod_preds$A_B2ii_DIREC,
                                                 meta_nets_cont, All_mod_preds, import_func_out_cont[[1]], 1:50, 'black', 
                                                 rowMeanCirc, 'blue')
B2ii_plot_approx_cont_2000_pop

B2ii_plot_approx_categ_100_nn1 <- directional_error_plots("B2ii", 100, All_mod_preds$A_B2ii_APPROX, All_mod_preds$A_B2ii_DIREC,
                                                          meta_nets_categ, All_mod_preds, import_func_out_categ[[1]], 1, 'blue',
                                                          rowMeanCirc)
B2ii_plot_approx_categ_100_nn1
B2ii_plot_approx_categ_100_nn2 <- directional_error_plots("B2ii", 100, All_mod_preds$A_B2ii_APPROX, All_mod_preds$A_B2ii_DIREC,
                                                          meta_nets_categ, All_mod_preds, import_func_out_categ[[1]], 2, 'blue',
                                                          rowMeanCirc)
B2ii_plot_approx_categ_100_nn2
B2ii_plot_approx_categ_100_pop <- directional_error_plots("B2ii", 100, All_mod_preds$A_B2ii_APPROX, All_mod_preds$A_B2ii_DIREC,
                                                  meta_nets_categ, All_mod_preds, import_func_out_categ[[1]], 1:50, 'black', 
                                                  rowMeanCirc, 'blue')
B2ii_plot_approx_categ_100_pop





#B2s

#png(filename="Write up/B2_BIG_signed_error_results.png", width=6.25, height=7.5,units="in",res=600)
plot_grid(B2i_plot_approx_cont_1000_nn1, B2i_plot_approx_cont_1000_nn2, B2i_plot_approx_cont_1000_pop, 
          B2i_plot_approx_categ_100_nn1, B2i_plot_approx_categ_100_nn2, B2i_plot_approx_categ_100_pop,
          B2ii_plot_approx_cont_2000_nn1, B2ii_plot_approx_cont_2000_nn2, B2ii_plot_approx_cont_2000_pop,
          B2ii_plot_approx_categ_100_nn1, B2ii_plot_approx_categ_100_nn2, B2ii_plot_approx_categ_100_pop,
          nrow=4, label_size=11, label_x = -0.16, label_y = 1.1, 
          labels = c('B2i/Continuous/1000points\n         Network 1', '\n         Network 2', '\n         Population',
                     'B2i/Categoric/100points\n         Network 1', '\n         Network 2', '\n         Population',
                     'B2ii/Continuous/2000points\n         Network 1', '\n         Network 2', '\n         Population',
                     'B2ii/Categoric/100points\n         Network 1', '\n         Network 2', '\n         Population'), 
          vjust=1.7, hjust=-0.2)
#dev.off()

png(filename="Write up/B2_signed_error_results_April24.png", width=7, height=6.5,units="in",res=600)
plot_grid(B2i_plot_approx_cont_1000_pop, 
          B2i_plot_approx_categ_100_pop,
          B2ii_plot_approx_cont_2000_pop,
          B2ii_plot_approx_categ_100_pop,
          nrow=2, label_size=11, label_x = 0, label_y = 1, 
          labels = c('B2i / Implementation 1 / 1000 points',
                     'B2i / Implementation 2 / 100 points',
                     'B2ii / Implementation 1 / 2000 points',
                     'B2ii / Implementation 2 / 100 points'), 
          vjust=1.7, hjust=-0.2)
dev.off()


#png(filename="Write up/B2_relative_results.png", width=6.5, height=6,units="in",res=600)
plot_grid(B2i_cont_plot2, B2i_categ_plot2, B2ii_cont_plot2, B2ii_categ_plot2, nrow=2,
          labels = c('B2i with Continuous Output', 'B2i with Categoric Output', 'B2ii with Continuous Output', 'B2ii with Categoric Output'), 
          vjust=1.7, hjust=-0.2)
#dev.off()

png(filename="Write up/B2_April24_results.png", width=6.5, height=6,units="in",res=600)
plot_grid(B2i_cont_plot3, B2i_categ_plot3, B2ii_cont_plot3, B2ii_categ_plot3, nrow=2,
          labels = c('B2i under Implementation 1', 'B2i under Implementation 2', 
                     'B2ii under Implementation 1', 'B2ii under Implementation 2'), 
          vjust=1.7, hjust=-0.2)
dev.off()

###### B3i

B3is_cont <- which(meta_nets_cont$grad2=="B3i")
B3i_means_cont <- nn_model_comparisons_absdiff_means_cont[nn_model_comparisons_absdiff_means_cont$grad2=="B3i",]

B3is_categ <- which(meta_nets_categ$grad2=="B3i")
B3i_means_categ <- nn_model_comparisons_absdiff_means_categ[nn_model_comparisons_absdiff_means_categ$grad2=="B3i",]

B3i_cont_lmms <- lmm_models_func("B3i", meta_nets_cont, B3i_means_cont, all_out_cont)
B3i_cont_lmms
B3i_categ_lmms <- lmm_models_func("B3i", meta_nets_categ, B3i_means_categ, all_out_categ)
B3i_categ_lmms

B3i_cont_sig_vs_T <- c(0, 0, 0, 2, 3, 3, 3, 3, 3)
B3i_categ_sig_vs_T <- c(0, 1, 3, 3, 3, 3, 3, 3, 0)

B3i_cont_sig_best_mod_comb <- c("APPROX", "APPROX", "APPROX", "APPROX", "APPROX", "APPROX")
B3i_cont_sig_best_mod_extrap <- c("R", "R", "R", "R", "R", "R")
B3i_categ_sig_best_mod_comb <- c("APPROX", "APPROX", "APPROX", "APPROX", "APPROX", "APPROX", "APPROX")
B3i_categ_sig_best_mod_extrap <- c("R", "R", "R", "R", "R", "R", "R")

B3i_cont_sig_vs_All <- c(0, 0, 0, 0, 0, 2, 3, 3, 2)
B3i_categ_sig_vs_All <- c(0, 0, 0, 0, 0, 0, 0, 0, 0)


B3i_cont_plot <- mod_error_plots(B3i_means_cont, c(B3i_means_cont$TRUE., 
                                                   B3i_means_cont$A_B3i_DIREC_T, B3i_means_cont$A_B3i_APPROX_T, B3i_means_cont$A_B3i_CORRECT_T,
                                                   B3i_means_cont$A_B3i_DIREC_R, B3i_means_cont$A_B3i_APPROX_R, B3i_means_cont$A_B3i_CORRECT_R,
                                                   B3i_means_cont$A_B3i_DIREC_Tr, B3i_means_cont$A_B3i_APPROX_Tr, B3i_means_cont$A_B3i_CORRECT_Tr),
                                 list("TRUE", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                 list("NA", "T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'none', 'none', B3i_cont_sig_vs_T, B3i_cont_sig_vs_All,
                                 T, B3i_cont_sig_best_mod_extrap, B3i_cont_sig_best_mod_comb)
B3i_cont_plot

B3i_cont_plot2 <- mod_error_plots2(B3i_means_cont, c(
                                                   B3i_means_cont$A_B3i_DIREC_T, B3i_means_cont$A_B3i_APPROX_T, B3i_means_cont$A_B3i_CORRECT_T,
                                                   B3i_means_cont$A_B3i_DIREC_R, B3i_means_cont$A_B3i_APPROX_R, B3i_means_cont$A_B3i_CORRECT_R,
                                                   B3i_means_cont$A_B3i_DIREC_Tr, B3i_means_cont$A_B3i_APPROX_Tr, B3i_means_cont$A_B3i_CORRECT_Tr),
                                 list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                 list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'legend', 'none', B3i_cont_sig_vs_T, B3i_cont_sig_vs_All,
                                 T, B3i_cont_sig_best_mod_extrap, B3i_cont_sig_best_mod_comb)
B3i_cont_plot2


B3i_cont_plot3 <- mod_error_plots3(B3i_means_cont, c(
  B3i_means_cont$A_B3i_DIREC_T, B3i_means_cont$A_B3i_APPROX_T, B3i_means_cont$A_B3i_CORRECT_T,
  B3i_means_cont$A_B3i_DIREC_R, B3i_means_cont$A_B3i_APPROX_R, B3i_means_cont$A_B3i_CORRECT_R,
  B3i_means_cont$A_B3i_DIREC_Tr, B3i_means_cont$A_B3i_APPROX_Tr, B3i_means_cont$A_B3i_CORRECT_Tr),
  list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
  list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'legend', 'legend', B3i_cont_sig_vs_T, B3i_cont_sig_vs_All)
B3i_cont_plot3

B3i_categ_plot <- mod_error_plots(B3i_means_categ, c(B3i_means_categ$TRUE., 
                                                    B3i_means_categ$A_B3i_DIREC_T, B3i_means_categ$A_B3i_APPROX_T, B3i_means_categ$A_B3i_CORRECT_T,
                                                    B3i_means_categ$A_B3i_DIREC_R, B3i_means_categ$A_B3i_APPROX_R, B3i_means_categ$A_B3i_CORRECT_R,
                                                    B3i_means_categ$A_B3i_DIREC_Tr, B3i_means_categ$A_B3i_APPROX_Tr, B3i_means_categ$A_B3i_CORRECT_Tr),
                                  list("TRUE", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                  list("NA", "T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'legend', 'none', B3i_categ_sig_vs_T, B3i_categ_sig_vs_All, 
                                  T, B3i_categ_sig_best_mod_extrap, B3i_categ_sig_best_mod_comb)
B3i_categ_plot

B3i_categ_plot2 <- mod_error_plots2(B3i_means_categ, c(
                                                     B3i_means_categ$A_B3i_DIREC_T, B3i_means_categ$A_B3i_APPROX_T, B3i_means_categ$A_B3i_CORRECT_T,
                                                     B3i_means_categ$A_B3i_DIREC_R, B3i_means_categ$A_B3i_APPROX_R, B3i_means_categ$A_B3i_CORRECT_R,
                                                     B3i_means_categ$A_B3i_DIREC_Tr, B3i_means_categ$A_B3i_APPROX_Tr, B3i_means_categ$A_B3i_CORRECT_Tr),
                                  list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                  list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'none', 'legend', B3i_categ_sig_vs_T, B3i_categ_sig_vs_All, 
                                  T, B3i_categ_sig_best_mod_extrap, B3i_categ_sig_best_mod_comb)
B3i_categ_plot2


B3i_categ_plot3 <- mod_error_plots3(B3i_means_categ, c(
  B3i_means_categ$A_B3i_DIREC_T, B3i_means_categ$A_B3i_APPROX_T, B3i_means_categ$A_B3i_CORRECT_T,
  B3i_means_categ$A_B3i_DIREC_R, B3i_means_categ$A_B3i_APPROX_R, B3i_means_categ$A_B3i_CORRECT_R,
  B3i_means_categ$A_B3i_DIREC_Tr, B3i_means_categ$A_B3i_APPROX_Tr, B3i_means_categ$A_B3i_CORRECT_Tr),
  list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
  list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'legend', 'legend', B3i_categ_sig_vs_T, B3i_categ_sig_vs_All)
B3i_categ_plot3



###### B3ii

B3iis_cont <- which(meta_nets_cont$grad2=="B3ii")
B3ii_means_cont <- nn_model_comparisons_absdiff_means_cont[nn_model_comparisons_absdiff_means_cont$grad2=="B3ii",]

B3iis_categ <- which(meta_nets_categ$grad2=="B3ii")
B3ii_means_categ <- nn_model_comparisons_absdiff_means_categ[nn_model_comparisons_absdiff_means_categ$grad2=="B3ii",]

B3ii_cont_lmms <- lmm_models_func("B3ii", meta_nets_cont, B3ii_means_cont, all_out_cont)
B3ii_cont_lmms
B3ii_categ_lmms <- lmm_models_func("B3ii", meta_nets_categ, B3ii_means_categ, all_out_categ)
B3ii_categ_lmms

B3ii_cont_sig_vs_T <- c(0, 0, 0, 2, 3, 3, 3, 0, 0)
B3ii_categ_sig_vs_T <- c(0, 3, 3, 3, 0, 0, 0, 0, 0)

B3ii_cont_sig_best_mod_comb <- c("APPROX", "APPROX", "APPROX", "APPROX")
B3ii_cont_sig_best_mod_extrap <- c("R", "R", "R", "R")
B3ii_categ_sig_best_mod_comb <- c("CORRECT", "APPROX", "APPROX")
B3ii_categ_sig_best_mod_extrap <- c("R", "R", "R")

B3ii_cont_sig_vs_All <- c(0, 0, 0, 0, 1, 0, 0, 0, 0)
B3ii_categ_sig_vs_All <- c(0, 0, 3, 0, 0, 0, 0, 0, 0)

B3ii_cont_plot <- mod_error_plots(B3ii_means_cont, c(B3ii_means_cont$TRUE., 
                                                   B3ii_means_cont$A_B3ii_DIREC_T, B3ii_means_cont$A_B3ii_APPROX_T, B3ii_means_cont$A_B3ii_CORRECT_T,
                                                   B3ii_means_cont$A_B3ii_DIREC_R, B3ii_means_cont$A_B3ii_APPROX_R, B3ii_means_cont$A_B3ii_CORRECT_R,
                                                   B3ii_means_cont$A_B3ii_DIREC_Tr, B3ii_means_cont$A_B3ii_APPROX_Tr, B3ii_means_cont$A_B3ii_CORRECT_Tr),
                                 list("TRUE", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                 list("NA", "T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'none', 'none', B3ii_cont_sig_vs_T, B3ii_cont_sig_vs_All,
                                 T, B3ii_cont_sig_best_mod_extrap, B3ii_cont_sig_best_mod_comb)
B3ii_cont_plot

B3ii_cont_plot2 <- mod_error_plots2(B3ii_means_cont, c(
                                                     B3ii_means_cont$A_B3ii_DIREC_T, B3ii_means_cont$A_B3ii_APPROX_T, B3ii_means_cont$A_B3ii_CORRECT_T,
                                                     B3ii_means_cont$A_B3ii_DIREC_R, B3ii_means_cont$A_B3ii_APPROX_R, B3ii_means_cont$A_B3ii_CORRECT_R,
                                                     B3ii_means_cont$A_B3ii_DIREC_Tr, B3ii_means_cont$A_B3ii_APPROX_Tr, B3ii_means_cont$A_B3ii_CORRECT_Tr),
                                  list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                  list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'none', 'legend', B3ii_cont_sig_vs_T, B3ii_cont_sig_vs_All,
                                  T, B3ii_cont_sig_best_mod_extrap, B3ii_cont_sig_best_mod_comb)
B3ii_cont_plot2

B3ii_cont_plot3 <- mod_error_plots3(B3ii_means_cont, c(
  B3ii_means_cont$A_B3ii_DIREC_T, B3ii_means_cont$A_B3ii_APPROX_T, B3ii_means_cont$A_B3ii_CORRECT_T,
  B3ii_means_cont$A_B3ii_DIREC_R, B3ii_means_cont$A_B3ii_APPROX_R, B3ii_means_cont$A_B3ii_CORRECT_R,
  B3ii_means_cont$A_B3ii_DIREC_Tr, B3ii_means_cont$A_B3ii_APPROX_Tr, B3ii_means_cont$A_B3ii_CORRECT_Tr),
  list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
  list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'legend', 'legend', B3ii_cont_sig_vs_T, B3ii_cont_sig_vs_All)
B3ii_cont_plot3

B3ii_categ_plot <- mod_error_plots(B3ii_means_categ, c(B3ii_means_categ$TRUE., 
                                                     B3ii_means_categ$A_B3ii_DIREC_T, B3ii_means_categ$A_B3ii_APPROX_T, B3ii_means_categ$A_B3ii_CORRECT_T,
                                                     B3ii_means_categ$A_B3ii_DIREC_R, B3ii_means_categ$A_B3ii_APPROX_R, B3ii_means_categ$A_B3ii_CORRECT_R,
                                                     B3ii_means_categ$A_B3ii_DIREC_Tr, B3ii_means_categ$A_B3ii_APPROX_Tr, B3ii_means_categ$A_B3ii_CORRECT_Tr),
                                  list("TRUE", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                  list("NA", "T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'none', 'legend', B3ii_categ_sig_vs_T, B3ii_categ_sig_vs_All,
                                  T, B3ii_categ_sig_best_mod_extrap, B3ii_categ_sig_best_mod_comb)
B3ii_categ_plot

B3ii_categ_plot2 <- mod_error_plots2(B3ii_means_categ, c(
                                                       B3ii_means_categ$A_B3ii_DIREC_T, B3ii_means_categ$A_B3ii_APPROX_T, B3ii_means_categ$A_B3ii_CORRECT_T,
                                                       B3ii_means_categ$A_B3ii_DIREC_R, B3ii_means_categ$A_B3ii_APPROX_R, B3ii_means_categ$A_B3ii_CORRECT_R,
                                                       B3ii_means_categ$A_B3ii_DIREC_Tr, B3ii_means_categ$A_B3ii_APPROX_Tr, B3ii_means_categ$A_B3ii_CORRECT_Tr),
                                   list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
                                   list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'legend', 'none', B3ii_categ_sig_vs_T, B3ii_categ_sig_vs_All,
                                   T, B3ii_categ_sig_best_mod_extrap, B3ii_categ_sig_best_mod_comb)
B3ii_categ_plot2

B3ii_categ_plot3 <- mod_error_plots3(B3ii_means_categ, c(
  B3ii_means_categ$A_B3ii_DIREC_T, B3ii_means_categ$A_B3ii_APPROX_T, B3ii_means_categ$A_B3ii_CORRECT_T,
  B3ii_means_categ$A_B3ii_DIREC_R, B3ii_means_categ$A_B3ii_APPROX_R, B3ii_means_categ$A_B3ii_CORRECT_R,
  B3ii_means_categ$A_B3ii_DIREC_Tr, B3ii_means_categ$A_B3ii_APPROX_Tr, B3ii_means_categ$A_B3ii_CORRECT_Tr),
  list("DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT", "DIREC", "APPROX", "CORRECT"),
  list("T", "T", "T", "R", "R", "R", "Tr", "Tr", "Tr"), 'legend', 'legend', B3ii_categ_sig_vs_T, B3ii_categ_sig_vs_All)
B3ii_categ_plot3

#B3s

#png(filename="Write up/B3_relative_results.png", width=6.5, height=6,units="in",res=600)
plot_grid(B3i_cont_plot2, B3i_categ_plot2, B3ii_cont_plot2, B3ii_categ_plot2, nrow=2,
          labels = c('B3i with Continuous Output', 'B3i with Categoric Output', 
                     'B3ii with Continuous Output', 'B3ii with Categoric Output'), vjust=1.7, hjust=-0.2)
#dev.off()

png(filename="Write up/B3_April24_results.png", width=7, height=9.5,units="in",res=600)
plot_grid(B3i_cont_plot3, B3i_categ_plot3, B3ii_cont_plot3, B3ii_categ_plot3, nrow=4,
          labels = c('B3i under Implementation 1', 'B3i under Implementation 2', 
                     'B3ii under Implementation 1', 'B3ii under Implementation 2'),
          vjust=1.7, hjust=-0.2)
dev.off()

###### B4i

B4is_cont <- which(meta_nets_cont$grad2=="B4i")
B4i_means_cont <- nn_model_comparisons_absdiff_means_cont[nn_model_comparisons_absdiff_means_cont$grad2=="B4i",]

B4is_categ <- which(meta_nets_categ$grad2=="B4i")
B4i_means_categ <- nn_model_comparisons_absdiff_means_categ[nn_model_comparisons_absdiff_means_categ$grad2=="B4i",]

B4i_cont_lmms <- lmm_models_func("B4i", meta_nets_cont, B4i_means_cont, all_out_cont)
B4i_cont_lmms
B4i_categ_lmms <- lmm_models_func("B4i", meta_nets_categ, B4i_means_categ, all_out_categ)
B4i_categ_lmms


B4i_cont_sig_vs_T <- c(0, 0, 0, 0, 3, 3, 3, 3, 3)
B4i_categ_sig_vs_T <- c(0, 0, 3, 3, 3, 3, 3, 3, 3)

B4i_cont_sig_best_mod_comb <- c("APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT")
B4i_cont_sig_best_mod_extrap <- c("Tr", "Tr", "Tr", "Tr", "Tr")
B4i_categ_sig_best_mod_comb <- c("APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT")
B4i_categ_sig_best_mod_extrap <- c("Tr", "Tr", "Tr", "Tr", "Tr", "Tr", "Tr")

B4i_cont_sig_vs_All <- c(0, 0, 0, 0, 3, 3, 3, 3, 3)
B4i_categ_sig_vs_All <- c(0, 0, 3, 3, 3, 3, 3, 3, 3)

B4i_cont_plot <- mod_error_plots(B4i_means_cont, c(B4i_means_cont$TRUE., B4i_means_cont$A_B4i_DIREC, 
                                                     B4i_means_cont$A_B4i_CORRECT.APPROX_T, B4i_means_cont$A_B4i_CORRECT.APPROX_R, 
                                                     B4i_means_cont$A_B4i_CORRECT.APPROX_Tr),
                                  list("TRUE", "DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                  list("NA", "NA", "T", "R", "Tr"), 'none', 'none', B4i_cont_sig_vs_T, B4i_cont_sig_vs_All,
                                 T, B4i_cont_sig_best_mod_extrap, B4i_cont_sig_best_mod_comb)
B4i_cont_plot

B4i_cont_plot2 <- mod_error_plots2(B4i_means_cont, c(B4i_means_cont$A_B4i_DIREC, 
                                                   B4i_means_cont$A_B4i_CORRECT.APPROX_T, B4i_means_cont$A_B4i_CORRECT.APPROX_R, 
                                                   B4i_means_cont$A_B4i_CORRECT.APPROX_Tr),
                                 list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                 list("NA", "T", "R", "Tr"), 'legend', 'none', B4i_cont_sig_vs_T, B4i_cont_sig_vs_All,
                                 T, B4i_cont_sig_best_mod_extrap, B4i_cont_sig_best_mod_comb)
B4i_cont_plot2

B4i_cont_plot3 <- mod_error_plots3(B4i_means_cont, c(B4i_means_cont$A_B4i_DIREC, 
                                                     B4i_means_cont$A_B4i_CORRECT.APPROX_T, B4i_means_cont$A_B4i_CORRECT.APPROX_R, 
                                                     B4i_means_cont$A_B4i_CORRECT.APPROX_Tr),
                                   list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                   list("NA", "T", "R", "Tr"), 'legend', 'legend', B4i_cont_sig_vs_T, B4i_cont_sig_vs_All)
B4i_cont_plot3

B4i_categ_plot <- mod_error_plots(B4i_means_categ, c(B4i_means_categ$TRUE., B4i_means_categ$A_B4i_DIREC, 
                                                     B4i_means_categ$A_B4i_CORRECT.APPROX_T, B4i_means_categ$A_B4i_CORRECT.APPROX_R, 
                                                     B4i_means_categ$A_B4i_CORRECT.APPROX_Tr),
                                  list("TRUE", "DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                  list("NA", "NA", "T", "R", "Tr"), 'legend', 'none', B4i_categ_sig_vs_T, B4i_categ_sig_vs_All,
                                  T, B4i_categ_sig_best_mod_extrap, B4i_categ_sig_best_mod_comb)
B4i_categ_plot

B4i_categ_plot2 <- mod_error_plots2(B4i_means_categ, c(B4i_means_categ$A_B4i_DIREC, 
                                                     B4i_means_categ$A_B4i_CORRECT.APPROX_T, B4i_means_categ$A_B4i_CORRECT.APPROX_R, 
                                                     B4i_means_categ$A_B4i_CORRECT.APPROX_Tr),
                                  list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                  list("NA", "T", "R", "Tr"), 'none', 'legend', B4i_categ_sig_vs_T, B4i_categ_sig_vs_All,
                                  T, B4i_categ_sig_best_mod_extrap, B4i_categ_sig_best_mod_comb)
B4i_categ_plot2

B4i_categ_plot3 <- mod_error_plots3(B4i_means_categ, c(B4i_means_categ$A_B4i_DIREC, 
                                                       B4i_means_categ$A_B4i_CORRECT.APPROX_T, B4i_means_categ$A_B4i_CORRECT.APPROX_R, 
                                                       B4i_means_categ$A_B4i_CORRECT.APPROX_Tr),
                                    list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                    list("NA", "T", "R", "Tr"), 'legend', 'legend', B4i_categ_sig_vs_T, B4i_categ_sig_vs_All)
B4i_categ_plot3


###### B4ii

B4iis_cont <- which(meta_nets_cont$grad2=="B4ii")
B4ii_means_cont <- nn_model_comparisons_absdiff_means_cont[nn_model_comparisons_absdiff_means_cont$grad2=="B4ii",]

B4iis_categ <- which(meta_nets_categ$grad2=="B4ii")
B4ii_means_categ <- nn_model_comparisons_absdiff_means_categ[nn_model_comparisons_absdiff_means_categ$grad2=="B4ii",]

B4ii_cont_lmms <- lmm_models_func("B4ii", meta_nets_cont, B4ii_means_cont, all_out_cont)
B4ii_cont_lmms
B4ii_categ_lmms <- lmm_models_func("B4ii", meta_nets_categ, B4ii_means_categ, all_out_categ)
B4ii_categ_lmms

B4ii_cont_sig_vs_TRUE <- c(0, 0, 0, 0, 3, 3, 3, 0, 0)
B4ii_categ_sig_vs_TRUE <- c(0, 0, 3, 3, 3, 0, 0, 0, 0)

B4ii_cont_sig_best_mod_comb <- c("APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT")
B4ii_cont_sig_best_mod_extrap <- c("Tr", "Tr", "Tr")
B4ii_categ_sig_best_mod_comb <- c("APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT")
B4ii_categ_sig_best_mod_extrap <- c("Tr", "Tr", "Tr")

B4ii_cont_sig_vs_All <- c(0, 0, 0, 0, 0, 1, 0, 0, 0)
B4ii_categ_sig_vs_All <- c(0, 0, 0, 0, 0, 0, 0, 0, 0)

B4ii_cont_plot <- mod_error_plots(B4ii_means_cont, c(B4ii_means_cont$TRUE., B4ii_means_cont$A_B4ii_DIREC, 
                                                   B4ii_means_cont$A_B4ii_CORRECT.APPROX_T, B4ii_means_cont$A_B4ii_CORRECT.APPROX_R, 
                                                   B4ii_means_cont$A_B4ii_CORRECT.APPROX_Tr),
                                 list("TRUE", "DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                 list("NA", "NA", "T", "R", "Tr"), 'none', 'none', B4ii_cont_sig_vs_TRUE, B4ii_cont_sig_vs_All,
                                 T, B4ii_cont_sig_best_mod_extrap, B4ii_cont_sig_best_mod_comb)
B4ii_cont_plot
B4ii_cont_plot2 <- mod_error_plots2(B4ii_means_cont, c(B4ii_means_cont$A_B4ii_DIREC, 
                                                     B4ii_means_cont$A_B4ii_CORRECT.APPROX_T, B4ii_means_cont$A_B4ii_CORRECT.APPROX_R, 
                                                     B4ii_means_cont$A_B4ii_CORRECT.APPROX_Tr),
                                  list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                  list("NA", "T", "R", "Tr"), 'none', 'legend', B4ii_cont_sig_vs_TRUE, B4ii_cont_sig_vs_All,
                                  T, B4ii_cont_sig_best_mod_extrap, B4ii_cont_sig_best_mod_comb)
B4ii_cont_plot2

B4ii_cont_plot3 <- mod_error_plots3(B4ii_means_cont, c(B4ii_means_cont$A_B4ii_DIREC, 
                                                       B4ii_means_cont$A_B4ii_CORRECT.APPROX_T, B4ii_means_cont$A_B4ii_CORRECT.APPROX_R, 
                                                       B4ii_means_cont$A_B4ii_CORRECT.APPROX_Tr),
                                    list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                    list("NA", "T", "R", "Tr"), 'legend', 'legend', B4ii_cont_sig_vs_TRUE, B4ii_cont_sig_vs_All)
B4ii_cont_plot3

B4ii_categ_plot <- mod_error_plots(B4ii_means_categ, c(B4ii_means_categ$TRUE., B4ii_means_categ$A_B4ii_DIREC, 
                                                     B4ii_means_categ$A_B4ii_CORRECT.APPROX_T, B4ii_means_categ$A_B4ii_CORRECT.APPROX_R, 
                                                     B4ii_means_categ$A_B4ii_CORRECT.APPROX_Tr),
                                  list("TRUE", "DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                  list("NA", "NA", "T", "R", "Tr"), 'none', 'legend', B4ii_categ_sig_vs_TRUE, B4ii_categ_sig_vs_All,
                                  T, B4ii_categ_sig_best_mod_extrap, B4ii_categ_sig_best_mod_comb)
B4ii_categ_plot

B4ii_categ_plot2 <- mod_error_plots2(B4ii_means_categ, c(B4ii_means_categ$A_B4ii_DIREC, 
                                                       B4ii_means_categ$A_B4ii_CORRECT.APPROX_T, B4ii_means_categ$A_B4ii_CORRECT.APPROX_R, 
                                                       B4ii_means_categ$A_B4ii_CORRECT.APPROX_Tr),
                                   list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                   list("NA", "T", "R", "Tr"), 'legend', 'none', B4ii_categ_sig_vs_TRUE, B4ii_categ_sig_vs_All,
                                   T, B4ii_categ_sig_best_mod_extrap, B4ii_categ_sig_best_mod_comb)
B4ii_categ_plot2

B4ii_categ_plot3 <- mod_error_plots3(B4ii_means_categ, c(B4ii_means_categ$A_B4ii_DIREC, 
                                                         B4ii_means_categ$A_B4ii_CORRECT.APPROX_T, B4ii_means_categ$A_B4ii_CORRECT.APPROX_R, 
                                                         B4ii_means_categ$A_B4ii_CORRECT.APPROX_Tr),
                                     list("DIREC", "APPROX/CORRECT", "APPROX/CORRECT", "APPROX/CORRECT"),
                                     list("NA", "T", "R", "Tr"), 'legend', 'legend', B4ii_categ_sig_vs_TRUE, B4ii_categ_sig_vs_All)
B4ii_categ_plot3


##B4s 

#png(filename="Write up/B4_relative_results.png", width=6.5, height=6,units="in",res=600)
plot_grid(B4i_cont_plot2, B4i_categ_plot2, B4ii_cont_plot2, B4ii_categ_plot2, nrow=2,
          labels = c('B4i with Continuous Output', 'B4i with Categoric Output',
                     'B4ii with Continuous Output', 'B4ii with Categoric Output'), vjust=1.7, hjust=-0.2)
#dev.off()

png(filename="Write up/B4_April24_results.png", width=7, height=6.5,units="in",res=600)
plot_grid(B4i_cont_plot3, B4i_categ_plot3, B4ii_cont_plot3, B4ii_categ_plot3, nrow=2,
          labels = c('B4i under Implementation 1', 'B4i under Implementation 2',
                     'B4ii under Implementation 1', 'B4ii under Implementation 2'), vjust=1.7, hjust=-0.2)
dev.off()


#Overall learning curves

cont_learning_curves <- data.frame(
  "Ys" = c(B1_means_cont$TRUE., B2i_means_cont$TRUE., B2ii_means_cont$TRUE., 
           B3i_means_cont$TRUE., B3ii_means_cont$TRUE., 
           B4i_means_cont$TRUE., B4ii_means_cont$TRUE.),
  "Xs" = c(B1_means_cont$Train_size, B2i_means_cont$Train_size, B2ii_means_cont$Train_size, 
           B3i_means_cont$Train_size, B3ii_means_cont$Train_size, 
           B4i_means_cont$Train_size, B4ii_means_cont$Train_size),
  "Envirs" = c(rep("B1", nrow(B1_means_cont)), rep("B2i", nrow(B2i_means_cont)), rep("B2ii", nrow(B2ii_means_cont)),
               rep("B3i", nrow(B3i_means_cont)), rep("B3ii", nrow(B3ii_means_cont)),
               rep("B4i", nrow(B4i_means_cont)), rep("B4ii", nrow(B4ii_means_cont)))
)


cont_learning_curves[which(cont_learning_curves$Xs==14000),]
end_train_cont_leaning_mods <- learning_mod_func(meta_nets_cont, all_out_cont, 14000)
end_train_cont_leaning_mods
which(end_train_cont_leaning_mods$p_value>0.05)

cont_lc_plot <- ggplot(data=cont_learning_curves)+
  theme_bw()+
  geom_line(aes(Xs, Ys, color=Envirs, linetype=Envirs), lwd=0.75)+
  geom_point(aes(Xs, Ys, color=Envirs, pch=Envirs), cex=2)+
  scale_y_continuous(breaks=c(0, pi/4, pi/2, 3*pi/4), limits=c(0, 3*pi/4),
                     labels=c("0", expression(pi ~ "/ 4"), expression(pi ~ "/ 2"), 
                              expression("3" ~ pi ~ "/ 4")))+
  scale_linetype_manual(name=NULL,values=c("B1"=1, "B2i"=3, "B2ii"=2, "B3i"=4, "B3ii"=5, "B4i"=6, "B4ii"=7), 
                        breaks=c("B1", "B2i", "B2ii","B3i", "B3ii", "B4i", "B4ii"), guide="none")+
  scale_shape_manual(name=NULL,values=c("B1"=15, "B2i"=16, "B2ii"=17, "B3i"=22, "B3ii"=6,"B4i"=23, "B4ii"=1),
                     breaks=c("B1", "B2i", "B2ii","B3i", "B3ii", "B4i", "B4ii"), guide="none")+
  scale_color_manual(name=NULL,values=c("B1"="red", "B2i"="purple", "B2ii"="blue", "B3i"="black", "B3ii"="darkgrey",
                                        "B4i"="green", "B4ii"="orange"),
                     breaks=c("B1", "B2i", "B2ii","B3i", "B3ii", "B4i", "B4ii"), guide="none")+
  theme(legend.position = c(0.7, 0.75), legend.box = "horizontal",
        panel.grid.minor = element_blank(), legend.key.width = unit(3, "line"))+ 
  ylab("Average absolute error (radians)")+
  xlab("Number of training datapoints")+
  theme(plot.margin = unit(c(1, 0.5, 0, 0), "cm"))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        legend.text=element_text(size=12))
cont_lc_plot

categ_learning_curves <- data.frame(
  "Ys" = c(B1_means_categ$TRUE., B2i_means_categ$TRUE., B2ii_means_categ$TRUE., 
           B3i_means_categ$TRUE., B3ii_means_categ$TRUE., 
           B4i_means_categ$TRUE., B4ii_means_categ$TRUE.),
  "Xs" = c(B1_means_categ$Train_size, B2i_means_categ$Train_size, B2ii_means_categ$Train_size, 
           B3i_means_categ$Train_size, B3ii_means_categ$Train_size, 
           B4i_means_categ$Train_size, B4ii_means_categ$Train_size),
  "Envirs" = c(rep("B1", nrow(B1_means_categ)), rep("B2i", nrow(B2i_means_categ)), rep("B2ii", nrow(B2ii_means_categ)),
               rep("B3i", nrow(B3i_means_categ)), rep("B3ii", nrow(B3ii_means_categ)),
               rep("B4i", nrow(B4i_means_categ)), rep("B4ii", nrow(B4ii_means_categ)))
)
categ_learning_curves[which(categ_learning_curves$Xs==14000),]
end_train_categ_leaning_mods <- learning_mod_func(meta_nets_categ, all_out_categ, 14000)
end_train_categ_leaning_mods
which(end_train_categ_leaning_mods$p_value>0.05)

categ_lc_plot <- ggplot(data=categ_learning_curves)+
  theme_bw()+
  geom_line(aes(Xs, Ys, color=Envirs, linetype=Envirs), lwd=0.75)+
  geom_point(aes(Xs, Ys, color=Envirs, pch=Envirs), cex=2)+
  scale_y_continuous(breaks=c(0, pi/4, pi/2, 3*pi/4), limits=c(0, 3*pi/4),
                     labels=c("0", expression(pi ~ "/ 4"), expression(pi ~ "/ 2"), 
                              expression("3" ~ pi ~ "/ 4")))+
  scale_linetype_manual(name=NULL,values=c("B1"=1, "B2i"=3, "B2ii"=2, "B3i"=4, "B3ii"=5, "B4i"=6, "B4ii"=7), 
                        breaks=c("B1", "B2i", "B2ii","B3i", "B3ii", "B4i", "B4ii"))+
  scale_shape_manual(name=NULL,values=c("B1"=15, "B2i"=16, "B2ii"=17, "B3i"=22, "B3ii"=6,"B4i"=23, "B4ii"=1),
                     breaks=c("B1", "B2i", "B2ii","B3i", "B3ii", "B4i", "B4ii"))+
  scale_color_manual(name=NULL,values=c("B1"="red", "B2i"="purple", "B2ii"="blue", "B3i"="black", "B3ii"="darkgrey",
                                        "B4i"="green", "B4ii"="orange"),
                     breaks=c("B1", "B2i", "B2ii","B3i", "B3ii", "B4i", "B4ii"))+
  theme(legend.position = c(0.7, 0.75), legend.box = "horizontal",
        panel.grid.minor = element_blank(), legend.key.width = unit(3, "line"))+ 
  ylab("Average absolute error (radians)")+
  xlab("Number of training datapoints")+
  theme(plot.margin = unit(c(1, 0.5, 0, 0), "cm"))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        legend.text=element_text(size=11))
categ_lc_plot

#png(filename="Write up/Overall_learning_curves.png", width=5.5, height=4,units="in",res=600)
plot_grid(cont_lc_plot, categ_lc_plot, nrow=1,
          labels = c('Continuous Output', 'Categoric Output'), vjust=1.7, hjust=-0.2)
#dev.off()



