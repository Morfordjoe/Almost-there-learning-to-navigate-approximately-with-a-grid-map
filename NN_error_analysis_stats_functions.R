library(vctrs)
library(lme4)
library(lmerTest)

import_process_func <- function(filenames_in){
  split_names <- unlist(strsplit(filenames_in, "/"))
  filenames_only <- split_names[seq(2, length(split_names), by=2)]
  meta_nets <- data.frame("names"=rep(NA, length(filenames_only)))
  meta_nets$names <- filenames_only
  meta_nets$grad1 <- as.factor(unlist(c(data.frame(strsplit(filenames_only, "_"))[2,])))
  meta_nets$grad2 <- as.factor(unlist(c(data.frame(strsplit(filenames_only, "_"))[3,])))
  meta_nets$Train_size <- as.integer(data.frame(strsplit(filenames_only, "_"))[5,])
  meta_nets$N_nets <- as.integer(data.frame(strsplit(filenames_only, "_"))[7,])
  
  order <- order(meta_nets$grad2, meta_nets$Train_size)
  meta_nets <-  meta_nets[order,]
  rownames(meta_nets) <- c(1:nrow(meta_nets))
  
  filenames_ordered <- filenames_in[order]
  
  ldf <- lapply(filenames_ordered, function(i){read.csv(i, header=F)})
  summary_ldf <- lapply(ldf, summary)
  names(summary_ldf) <- data.frame(strsplit(filenames_ordered, "/"))[2,]
  names(ldf) <- names(summary_ldf)
  
  meta_nets <- data.frame("names"=names(summary_ldf))
  meta_nets$grad1 <- as.factor(unlist(c(data.frame(strsplit(names(summary_ldf), "_"))[2,])))
  meta_nets$grad2 <- as.factor(unlist(c(data.frame(strsplit(names(summary_ldf), "_"))[3,])))
  meta_nets$Train_size <- as.integer(data.frame(strsplit(names(summary_ldf), "_"))[5,])
  meta_nets$N_nets <- as.integer(data.frame(strsplit(names(summary_ldf), "_"))[7,])
  
  return(list(ldf, meta_nets))
}


abs_error_function <- function(ldf_in, meta_nets_in, All_mod_preds_in, B1_mods_in, B2i_mods_in, B2ii_mods_in, 
                               B3i_mods_in, B3ii_mods_in, B4i_mods_in, B4ii_mods_in){
  
  nn_model_comparisons_absdiff_means <- data.frame(matrix(nrow=nrow(meta_nets_in), ncol=ncol(All_mod_preds_in)))
  colnames(nn_model_comparisons_absdiff_means) <- colnames(All_mod_preds_in)
  nn_model_comparisons_absdiff_means$grad2 <- meta_nets_in$grad2
  nn_model_comparisons_absdiff_means$Train_size <- meta_nets_in$Train_size
  
  all_out <- list()
  for (rw in 1:
       nrow(meta_nets_in)){
    print(rw)
    nns_out <- ldf_in[[rw]]
    if(meta_nets_in$grad2[rw] == "B1"){
      test_mods <- c(1, B1_mods_in)
    }else if(meta_nets_in$grad2[rw] == "B2i"){
      test_mods <- c(1, B2i_mods_in)
    }else if(meta_nets_in$grad2[rw] == "B2ii"){
      test_mods <- c(1, B2ii_mods_in)
    }else if(meta_nets_in$grad2[rw] == "B3i"){
      test_mods <- c(1, B3i_mods_in)
    }else if(meta_nets_in$grad2[rw] == "B3ii"){
      test_mods <- c(1, B3ii_mods_in)
    }else if(meta_nets_in$grad2[rw] == "B4i"){
      test_mods <- c(1, B4i_mods_in)
    }else if(meta_nets_in$grad2[rw] == "B4ii"){
      test_mods <- c(1, B4ii_mods_in)
    }
    nrows <- meta_nets_in$N_nets[rw]*length(test_mods)
    all_out[[rw]] <- data.frame("mean_each_abs_errors" = rep(NA, nrows), 
                                "Network_n" = as.factor(rep(c(1:meta_nets_in$N_nets[rw]), length(test_mods))),
                                "t_mod" = as.factor(rep(colnames(All_mod_preds)[test_mods],
                                                        each=meta_nets_in$N_nets[rw])))
    for (tm in 1:length(test_mods)){
      t_mod <- test_mods[tm]
      
      errors <- nns_out - All_mod_preds_in[,t_mod]
      errors[errors>pi] <- errors[errors>pi] - 2*pi
      
      abs_errors <- abs(errors)
      mean_each_abs_errors <- colMeans(abs_errors)
      all_out[[rw]]$mean_each_abs_errors[all_out[[rw]]$t_mod==colnames(All_mod_preds)[t_mod]] <- mean_each_abs_errors
      
      nn_model_comparisons_absdiff_means[rw, t_mod] <- mean(mean_each_abs_errors)
    }
  }
  return(list(all_out, nn_model_comparisons_absdiff_means))
}



rowMeanCirc <- function(data) apply(data, 1, function(x) suppressWarnings(mean.circular(x)))
colMeanCirc <- function(data) apply(data, 2, function(x) suppressWarnings(mean.circular(x)))


lmm_models_func <- function(grad_in, meta_nets_in, means_in, all_out_in){
  lmms <- list()
  inds <- which(meta_nets_in$grad2==grad_in)
  meta_nets_in_subset <- meta_nets_in[inds,]
  
  for (i in 1:nrow(meta_nets_in_subset)){
    best_mod <- colnames(means_in)[which(means_in[i,]==min(means_in[i,(1:(ncol(means_in)-2))], na.rm=T))]
    all_out_in[[inds[i]]]$t_mod2 <- relevel(all_out_in[[inds[i]]]$t_mod, ref=best_mod)
    mod <- lmer(mean_each_abs_errors~t_mod2+(1|Network_n), data=all_out_in[[inds[i]]])
    lmms[[i]] <- list(meta_nets_in_subset[i,],
                         best_mod,
                         mod,
                         summary(mod))
  }
  return(lmms)
}


mod_error_plots <- function(mod_means_in, vec_mods_in, list_mod_names_in, list_extrap_names_in, lty_legend_or_none='legend', col_legend_or_none = 'legend', 
                            sig_vec_vs_T=rep(0, 9), sig_vec_vs_All=rep(0, 9), 
                            plot_best_mod_symbol=F, best_mod_col=rep(NA, 9), best_mod_symbol=rep(NA, 9)){
  
  plotting_df <- data.frame(vec_mods_in, rep(mod_means_in$Train_size, times=length(list_mod_names_in)),
                            unlist(rep(list_mod_names_in, each=nrow(mod_means_in))), 
                            unlist(rep(list_extrap_names_in, each=nrow(mod_means_in))))
  names(plotting_df) <- c("ys", "train_ns", "mods", "extrap")

  x_stars <- vec_rep_each(mod_means_in$Train_size, sig_vec_vs_T)
  star_ys <- c(0.14, 0.105, 0.075)
  y_stars <- unlist(lapply(sig_vec_vs_T, function(i) head(star_ys, i)))
  stars <- data.frame(x_stars, y_stars)
  
  x_stars_all <- vec_rep_each(mod_means_in$Train_size, sig_vec_vs_All)
  star_ys_all <- c(0.045, 0.025, 0.01)
  y_stars_all <- unlist(lapply(sig_vec_vs_All, function(i) head(star_ys_all, i)))
  stars_all <- data.frame(x_stars_all, y_stars_all)
  
  plot_out <- ggplot(data=plotting_df)+
    geom_point(aes(train_ns, ys, pch=mods, col=extrap))+
    geom_line(aes(train_ns, ys, linetype=mods, col=extrap))+
    geom_point(data=stars, aes(x_stars, y_stars), pch=8)+
    geom_point(data=stars_all, aes(x_stars_all, y_stars_all), pch=8, col="darkgoldenrod1")+
    scale_x_sqrt(breaks=mod_means_in$Train_size)+
    scale_y_sqrt(breaks=c(0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5), limits=c(0, 3*pi/4))+
    scale_linetype_manual(name=NULL,values=c("TRUE"=1, "DIREC"=3, "APPROX"=2, "CORRECT"=4, "APPROX/CORRECT"=6),
                          breaks=c("TRUE", "APPROX/CORRECT", "APPROX","DIREC", "CORRECT"), guide=lty_legend_or_none)+
    scale_shape_manual(name=NULL,values=c("TRUE"=15, "DIREC"=16, "APPROX"=17, "CORRECT"=22, "APPROX/CORRECT"=6),
                       breaks=c("TRUE", "APPROX/CORRECT", "APPROX","DIREC", "CORRECT"), guide=lty_legend_or_none)+
    scale_color_manual(name=NULL,values=c("T"="red", "R"="purple", "Tr"="blue", "NA"="black"),
                       breaks=c("T","R","Tr", "NA"), guide=col_legend_or_none)+
    theme(legend.position = c(0.7, 0.92), legend.box = "horizontal",
          panel.grid.minor = element_blank(), legend.key.width = unit(1.5, "line"))+ 
    ylab("Average absolute error (radians)")+
    xlab("Number of training datapoints")+
    theme(plot.margin = unit(c(1, 0.5, 0, 0), "cm"))+
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
          legend.text=element_text(size=7))
  
  if (lty_legend_or_none=='legend'){
    plot_out <- plot_out +guides(linetype = guide_legend(nrow=2,byrow=TRUE, override.aes = list(size=1.5, col="darkgreen")))
  }
  if (col_legend_or_none=='legend'){
    plot_out <- plot_out +guides(color = guide_legend(nrow=2,byrow=TRUE, override.aes = list(pch=18)))
  }
  if (plot_best_mod_symbol==T){
    star_xs_unique <- unique(x_stars)
    ys <- rep(0, length(star_xs_unique))
    best_mod_df <- data.frame(star_xs_unique, ys, best_mod_symbol, best_mod_col)
    plot_out <- plot_out + geom_point(data=best_mod_df, aes(star_xs_unique, ys, pch=best_mod_symbol, col=best_mod_col), cex=2)
  }
  
  return(plot_out)
  
}


mod_error_plots2 <- function(mod_means_in, vec_mods_in, list_mod_names_in, list_extrap_names_in, pch_legend_or_none='legend', col_legend_or_none = 'legend', 
                            sig_vec_vs_T=rep(0, 9), sig_vec_vs_All=rep(0, 9), 
                            plot_best_mod_symbol=F, best_mod_col=rep(NA, 9), best_mod_symbol=rep(NA, 9)){
  
  mods_in_rel_TRUE <- vec_mods_in - mod_means_in[,1]
  plotting_df <- data.frame(mods_in_rel_TRUE, rep(mod_means_in$Train_size, times=length(list_mod_names_in)),
                            unlist(rep(list_mod_names_in, each=nrow(mod_means_in))), 
                            unlist(rep(list_extrap_names_in, each=nrow(mod_means_in))))
  names(plotting_df) <- c("ys", "train_ns", "mods", "extrap")
  
  x_stars <- vec_rep_each(mod_means_in$Train_size, sig_vec_vs_T)
  star_ys <- c(-0.4, -0.425, -0.45)
  y_stars <- unlist(lapply(sig_vec_vs_T, function(i) head(star_ys, i)))
  stars <- data.frame(x_stars, y_stars)
  
  x_stars_all <- vec_rep_each(mod_means_in$Train_size, sig_vec_vs_All)
  star_ys_all <- c(-0.475, -0.5, -0.525)
  y_stars_all <- unlist(lapply(sig_vec_vs_All, function(i) head(star_ys_all, i)))
  stars_all <- data.frame(x_stars_all, y_stars_all)
  
  plot_out <- ggplot(data=plotting_df)+
    geom_abline(intercept=0, slope=0)+
    geom_point(aes(train_ns, ys, pch=mods, col=extrap))+
    geom_line(aes(train_ns, ys, group=interaction(mods, extrap), linetype=extrap, col=extrap))+
    geom_point(data=stars, aes(x_stars, y_stars), pch=8, size=0.8)+
    geom_point(data=stars_all, aes(x_stars_all, y_stars_all), pch=8, col="darkgoldenrod1", size=0.8)+
    scale_x_sqrt(breaks=mod_means_in$Train_size)+
    coord_cartesian(ylim=c(-0.55, 0.15))+
    scale_shape_manual(name=NULL,values=c("TRUE"=15, "DIREC"=16, "APPROX"=17, "CORRECT"=22, "APPROX/CORRECT"=6),
                       breaks=c("TRUE", "APPROX/CORRECT", "APPROX","DIREC", "CORRECT"), guide=pch_legend_or_none)+
    scale_color_manual(name=NULL,values=c("T"="red", "R"="purple", "Tr"="blue", "NA"="black"),
                       breaks=c("T","R","Tr", "NA"), guide=col_legend_or_none)+
    scale_linetype_manual(name=NULL,values=c("T"=1, "R"=2, "Tr"=4, "NA"=3),
                          breaks=c("T","R","Tr", "NA"), guide=col_legend_or_none)+
    theme(legend.position = "bottom", legend.margin=margin(t = 0, unit='cm'), legend.box = "vertical",
          panel.grid.minor = element_blank(), legend.key.width = unit(1, "line"))+ 
    ylab("Model fit (radians)")+
    xlab("Number of training datapoints")+
    theme(plot.margin = unit(c(1, 0.5, 0, 0), "cm"))+
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
          legend.text=element_text(size=7))
  
  if (pch_legend_or_none=='legend'){
    legend_rows <- ifelse(length(list_mod_names_in)>1, 2, 1)
    plot_out <- plot_out +guides(shape = guide_legend(nrow=1, byrow=TRUE, override.aes = list(size=1.5, col="darkgreen")))
  }
  if (col_legend_or_none=='legend'){
    legend_rows2 <- ifelse(length(list_extrap_names_in)>1, 2, 1)
    plot_out <- plot_out +guides(color = guide_legend(nrow=1, byrow=TRUE, override.aes = list(shape=NA)))
  }
  if (plot_best_mod_symbol==T){
    star_xs_unique <- unique(x_stars)
    ys <- rep(-0.55, length(star_xs_unique))
    best_mod_df <- data.frame(star_xs_unique, ys, best_mod_symbol, best_mod_col)
    plot_out <- plot_out + geom_point(data=best_mod_df, aes(star_xs_unique, ys, pch=best_mod_symbol, col=best_mod_col), size=1.25)
  }
  
  return(plot_out)
  
}


directional_error_plots <- function(grad, size, approx_mod_in, direc_mod_in,
                                    meta_nets_in, All_mod_preds_in, ldf, which_nns, point_col, rowMeanCirc_func,
                                    predic_col = "red"){
  
  ind <- which(meta_nets_in$grad2==grad & meta_nets_in$Train_size==size)

  all_data <- ldf[[ind]]
  if (length(which_nns)>1){
    y <- rowMeanCirc_func(all_data[,which_nns])
  }else{
    y <- all_data[,which_nns]
  }
  y_error <- y - All_mod_preds_in$TRUE.
  y_error[y_error>pi] <- y_error[y_error>pi] - 2*pi
  y_error[y_error<(-pi)] <- y_error[y_error<(-pi)] + 2*pi
  
  approx_pred <- approx_mod_in
  approx_pred_error <- approx_pred - All_mod_preds_in$TRUE.
  approx_pred_error[approx_pred_error>pi] <- approx_pred_error[approx_pred_error>pi] - 2*pi
  approx_pred_error[approx_pred_error<(-pi)] <- approx_pred_error[approx_pred_error<(-pi)] + 2*pi
  
  direc_pred <- direc_mod_in
  direc_pred_error <- direc_pred - All_mod_preds_in$TRUE.
  direc_pred_error[direc_pred_error>pi] <- direc_pred_error[direc_pred_error>pi] - 2*pi
  direc_pred_error[direc_pred_error<(-pi)] <- direc_pred_error[direc_pred_error<(-pi)] + 2*pi

  order_inds <- order(All_mod_preds_in$TRUE.)
    
  #plot(y_error~All_mod_preds_in$TRUE., ylim=c(-2, 2), pch=16, cex=0.2)
  #abline(h=0)
  #lines(approx_pred_error[order_inds]~All_mod_preds_in$TRUE.[order_inds], col="red", lwd=2.5, lty="dashed")
  
  plot_out <- ggplot()+
    theme_bw()+
    geom_point(aes(All_mod_preds_in$TRUE., y_error, pch=mods), pch=16, cex=0.2, col=point_col)+
    #geom_line(aes(All_mod_preds_in$TRUE.[order_inds], direc_pred_error[order_inds]), col='blue', lwd=0.5)+
    geom_line(aes(All_mod_preds_in$TRUE.[order_inds], approx_pred_error[order_inds]), col=predic_col, lwd=1)+
    xlab("Test direction (radians)")+
    ylab("Error (radians)")+
    theme(plot.margin = unit(c(1, 0.5, 0, 0), "cm"))+
    scale_y_continuous(breaks=c(-pi, -pi/2, 0, pi/2, pi), limits=c(-pi, pi),expand = expansion(),
                       labels=c(expression("-" ~ pi), expression("-" ~ pi ~ "/ 2"), "0", 
                                expression(pi ~ "/ 2"), expression(pi)))+
    scale_x_continuous(breaks=c(-pi, -pi/2, 0, pi/2, pi), limits=c(-pi, pi),expand = expansion(),
                       labels=c(expression("-" ~ pi), expression("-" ~ pi ~ "/ 2"), "0", 
                                expression(pi ~ "/ 2"), expression(pi)))
  return(plot_out)
}


mod_error_plots3 <- function(mod_means_in, vec_mods_in, list_mod_names_in, list_extrap_names_in, col_legend_or_none='legend', shape_legend_or_none = 'legend', 
                             sig_vec_vs_T=rep(0, 9), sig_vec_vs_All=rep(0, 9)){
  
  mods_in_rel_TRUE <- vec_mods_in - mod_means_in[,1]
  plotting_df <- data.frame(vec_mods_in, rep(mod_means_in$Train_size, times=length(list_mod_names_in)),
                            as.factor(unlist(rep(list_mod_names_in, each=nrow(mod_means_in)))), 
                            as.factor(unlist(rep(list_extrap_names_in, each=nrow(mod_means_in)))),
                            rep(mod_means_in[,1], length(list_mod_names_in)))
  names(plotting_df) <- c("ys", "train_ns", "mods", "extrap", "baseline")
  
  x_stars <- vec_rep_each(c(1:9), sig_vec_vs_T)
  star_xs_add <- c(0, -0.25, 0.25)
  x_adds <- unlist(lapply(sig_vec_vs_T, function(i) head(star_xs_add, i)))
  x_stars <- x_stars + x_adds
  y_stars <- rep(0.1, length(x_stars))
  stars <- data.frame(x_stars, y_stars)
  
  x_stars_all <- vec_rep_each(c(1:9), sig_vec_vs_All)
  x_stars_add_all <- c(0, -0.25, 0.25)
  x_stars_all_adds <- unlist(lapply(sig_vec_vs_All, function(i) head(x_stars_add_all, i)))
  x_stars_all <- x_stars_all + x_stars_all_adds
  y_stars_all <- rep(0.025, length(x_stars_all))
  stars_all <- data.frame(x_stars_all, y_stars_all)
  
  plot_out <- ggplot()+
    geom_segment(aes(x=c((1:9)-0.5), xend=c((1:9)+0.5),
                     y=unique(mod_means_in[,1]), yend=unique(mod_means_in[,1])),
                 linetype="dashed")+
    #geom_segment(data=plotting_df, aes(x = -0.05 + as.numeric(as.factor(train_ns)) + 9*rep(seq(-0.05*(length(list_mod_names_in)-1), 0.05*(length(list_mod_names_in)-1), length.out=length(list_mod_names_in)), each=9)/length(list_mod_names_in),
          #                             xend = 0.05 + as.numeric(as.factor(train_ns)) + 9*rep(seq(-0.05*(length(list_mod_names_in)-1), 0.05*(length(list_mod_names_in)-1), length.out=length(list_mod_names_in)), each=9)/length(list_mod_names_in),
          #                             y = baseline, 
           #                            col=mods, group=extrap),linewidth=1.2)+
    geom_segment(data=plotting_df, aes(x = as.numeric(as.factor(train_ns)) + 9*rep(seq(-0.05*(length(list_mod_names_in)-1), 0.05*(length(list_mod_names_in)-1), length.out=length(list_mod_names_in)), each=9)/length(list_mod_names_in),
                                       y = ys, 
                                       col=mods, group=extrap, yend=baseline+0.01),linewidth=2)+
    geom_segment(data=plotting_df, aes(x = as.numeric(as.factor(train_ns)) + 9*rep(seq(-0.05*(length(list_mod_names_in)-1), 0.05*(length(list_mod_names_in)-1), length.out=length(list_mod_names_in)), each=9)/length(list_mod_names_in),
                                       y = ys, 
                                       col=mods, group=extrap, yend=baseline-0.01),linewidth=2)+
    geom_point(data=plotting_df, cex=1.5, aes(x = as.numeric(as.factor(train_ns)) + 9*rep(seq(-0.05*(length(list_mod_names_in)-1), 0.05*(length(list_mod_names_in)-1), length.out=length(list_mod_names_in)), each=9)/length(list_mod_names_in),
                                            y = ys, fill=mods,
                                            group=extrap),  color="white", shape=21)+
    geom_point(data=plotting_df, aes(x = as.numeric(as.factor(train_ns)) + 9*rep(seq(-0.05*(length(list_mod_names_in)-1), 0.05*(length(list_mod_names_in)-1), length.out=length(list_mod_names_in)), each=9)/length(list_mod_names_in),
                                                  y = ys, fill=extrap,
                                                  group=extrap, shape=extrap), color="white", cex=2)+
    geom_point(data=stars, aes(x_stars, y_stars), pch=8, cex=1.5)+
    geom_point(data=stars_all, aes(x_stars_all, y_stars_all), pch=8, col="darkgoldenrod1", cex=1.5)+
    scale_x_continuous(breaks=seq(1, 9, by=1),
                       labels=mod_means_in$Train_size)+
    scale_y_continuous(limits=c(0, 5*pi/8), breaks=c(0, pi/4, pi/2, 3*pi/4),
                       labels=c("0", expression(~ pi ~ "/ 4"), expression(~ pi ~ "/ 2"), 
                                expression("3" ~ pi ~ "/ 4")))+
    scale_shape_manual(name=NULL,values=c("T"=24, "R"=22, "Tr"=23, "NA"=NA),
                       breaks=c("T","R","Tr"), guide=shape_legend_or_none,
                       labels=c("T","R","Tr"))+
    scale_fill_manual(name=NULL,values=c("DIREC"="pink", "APPROX/CORRECT"="purple",  "APPROX"="blue",  "CORRECT"="red",
                                         "T"="black", "R"="grey", "Tr"="green", "NA"=NA),
                      breaks=c("T","R","Tr"), guide=shape_legend_or_none,
                      labels=c("T","R","Tr"))+
    scale_color_manual(name=NULL,values=c("DIREC"="pink", "APPROX/CORRECT"="purple", "APPROX"="blue", "CORRECT"="red",
                                          "T"="black", "R"="grey", "Tr"="green", "NA"=NA),
                       breaks=c("DIREC", "APPROX/CORRECT", "APPROX", "CORRECT"), 
                       labels=c("DIREC", "APPROX/\nCORRECT", "APPROX", "CORRECT"),
                       guide=col_legend_or_none)+
    theme_bw()+
    theme(legend.position = c(0.5, 0.93), legend.box = "horizontal",
          panel.grid.minor = element_blank(), legend.key.width = unit(1, "line"),
          legend.text=element_text(size=6.5), legend.direction="horizontal",
          legend.background = element_rect(fill="lightblue"),
          plot.margin = unit(c(1, 0.5, 0, 0), "cm"), axis.text.x = element_text(size=7))+ 
    ylab("Average absolute error (radians)")+
    xlab("Number of training datapoints")+
    geom_vline(xintercept = (0:9)+0.5)

  
  return(plot_out)
  
}

learning_mod_func <- function(meta_nets_in, all_out_in, n_datapoint_test){
  train_point_meta <- which(meta_nets_in$Train_size==n_datapoint_test)
  envs_train_point <- meta_nets_in$grad2[train_point_meta]

  learning_df <- data.frame()
  for (i in 1:length(train_point_meta)){
    mean_network_results <- all_out_in[[train_point_meta[i]]]
    mean_network_results_trueonly <- mean_network_results[which(mean_network_results$t_mod=="TRUE."),]
    env <- envs_train_point[i]
    to_return_loop_data <- data.frame(mean_network_results_trueonly$mean_each_abs_errors, rep(env, nrow(mean_network_results_trueonly)))
    learning_df <- rbind(learning_df, to_return_loop_data)
  }
  colnames(learning_df) <- c("mean_each_abs_errors", "env")

  all_pair_t_tests_out <- data.frame()
  for (i in 1:(length(envs_train_point)-1)){
    env_i <- envs_train_point[i]
    env_i_results <- learning_df$mean_each_abs_errors[which(learning_df$env==env_i)]
    for (j in (i+1):length(envs_train_point)){
      env_j <- envs_train_point[j]
      env_j_results <- learning_df$mean_each_abs_errors[which(learning_df$env==env_j)]
      t_test_out <- t.test(env_i_results, env_j_results)
      all_pair_t_tests_out <- rbind(all_pair_t_tests_out, data.frame(env_i, env_j, t_test_out$p.value))
    }
  }
  colnames(all_pair_t_tests_out) <- c("env1", "env2", "p_value")
  return(all_pair_t_tests_out)
}
  
  
  