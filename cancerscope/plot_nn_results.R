#THIS SCRIPT TAKES IN A CONFIDENCE PREDICTION FILE FILE FOR 1 OR MORE SAMPLES
#AND PLOTS A GRAPH SHOWING CONFIDENCE FOR EVERY DISEASE
#AS ORDERED BY THE SPEARMAN CORRELATION PLOT, AND TISSUE ANNOTATIONS

##Get required libs
lapply(c("stringr", "reshape2", "plyr","ggplot2","scales","ggrepel"), require, character.only = TRUE)

#Define Plotting Defaults
plot_defaults=theme_bw(base_size = 18)+ theme(axis.text.x = element_text(angle=45,hjust=1)) 
disease_order=c("MB-Adult","GBM","TFRI_GBM_NCL","LGG","THCA","FL","DLBC","NCI_GPH_DLBCL","DLBC_BM","THYM","LAML","UVM","SKCM","MESO","LUAD","BRCA","PCPG","ACC","KIRC","KIRP","KICH","LIHC","PAAD","STAD","COADREAD","ESCA_EAC","CHOL","TGCT","BLCA","PRAD","SARC","OV","UCS","UCEC","CESC_CAD","HNSC","LUSC","ESCA_SCC","CESC_SCC")

#Get input file with confidence per class, predicted from NN. Passed as input.
args = commandArgs(trailingOnly=TRUE)
if (length(args)!=5){
	stop("Five arguments needed - trained models list, dir with prediction sets for all models, pattern to subset dirs by, output dir for plots (main), prefix for plots", call.=FALSE)
} else {
	modelsource=args[1] 
	indir=args[2]	
	whichSample= args[3]
	plotdir=args[4]
	plotprefix=args[5]
}

base_functions="stats_plot.R"
source(base_functions)

#Define default files for disease list (used to plot graph)
diseaselist_file = "cancerscope/resources/custom_ped_pog_diseasetypes_v8.txt"
if ( !(file.exists(plotdir))) {
    dir.create(file.path(plotdir))
    cat("Generating Plots in Directory: ", plotdir)
    cat ("Press [enter] to continue")
    line <- readline()
}

#Source required variables and functions
source(modelsource, echo=TRUE)
dat_classize = data.frame()
v1_models = c("none17k","rm500","rm500dropout","none17kdropout", "smotenone17k")
num_models = 5
for(i in v1_models){
  print(i)
  qdir = eval(as.symbol(i))
  sizefile = list.files(path=qdir, pattern="dict_labcount", recursive=TRUE, full.names=TRUE)
  tdat = read.table(sizefile[length(sizefile)]); tdat$model = i
  dat_classize = rbind(tdat,dat_classize)
}
if(dim(unique(dat_classize[,c("V1","V2")]))[1] == dim(dat_classize)[1]/length(v1_models)){
  dat_classize = unique(dat_classize[,c("V1","V2")])
}else{
  print("WARNING: Different training class sizes, cannot collapse model training sets!")
}
colnames(dat_classize) = c("dis","class_count")
dat_classize$disease=gsub("_[N,T]S$","",dat_classize$dis)
dat_classize$type = gsub("[A-Z,-,a-z,MB-]*_","",dat_classize$dis)

#Get the pattern matching directories, and then the prediction_values_all.txt file in each
indirs=list.files(path=indir, pattern="prediction_values_all", recursive=TRUE, full.names = TRUE)
indirs=indirs[grep(whichSample, indirs)] #Only keep the entries corresponding to the POG under consideration
dat2=data.frame(); dat_full=data.frame(); 
for(i in indirs){
  dat<-read.table(i,check.names=FALSE,sep="\t", header=TRUE, stringsAsFactors = FALSE); dat$file=i;
  #dat$file = str_match(dat$file, "TESTPRED_.*/"); 
  dat_full=rbind(dat,dat_full)
  dat2 <- rbind(dat,dat2)
}
dat2$Biopsy = dat2$sample;
dat2$sample = paste(whichSample, dat2$sample, sep="_")

#Only subselect the entries corresponding to the current POG
Sampledat = dat2[grep(whichSample, dat2$sample),] #
Sampledat = dat2[grep(whichSample, dat2$file),]
bestcalldat = Sampledat[,c("predicted","confidence","sample","file")]; 
mostvoted_class = get_most_voted(bestcalldat)
sample_nn_all = Sampledat;
sample_nn_all$disease=gsub("_[N,T]S$","",sample_nn_all$predicted)
sample_nn_all$type = gsub("[A-Z,-,a-z,MB-]*_","",sample_nn_all$predicted)
dat3=merge(sample_nn_all,dis_type, by="disease")
plot_t=list(); plot_tmean=list() ; plot_all=list(); plot_all_nt=list(); plot_all_ntscale=list(); plot_trange=list()
i=1

for(smpl in unique(dat3$sample)){
  #Subset sample specific predictions
  sampdat=dat3[dat3$sample==smpl,]
  biop_tc = str_match(smpl, "_.*_([A-Z,0-9]*)")[2]
  smpl_tc = NA
  if(is.na(smpl_tc)){smpl_tc=0}else{smpl_tc=smpl_tc/100}
  #Build plot dataframe
  sampdat_nn=melt(sampdat[,!names(sampdat) %in% c("sample","Source","Biopsy","predicted","confidence")], id=c("file","disease","type","actual")); 
  sampdat_nn$actual_type = gsub("[A-Z,-,a-z,MB-]*_","",sampdat_nn$actual)
  sampdat_nn$variable_type = gsub("[A-Z,-,a-z,MB-]*_","",sampdat_nn$variable)
  sampdat_nn$actual_dis= sampdat_nn$actual
  sampdat_nn$variable_dis = gsub("_[N,T]S$","",sampdat_nn$variable)
  
  sampdat_nn$disease <- factor(sampdat_nn$disease, levels=disease_order); 
  colnames(sampdat_nn) <- c("model","predicted_disease","pred_type","actual","dis","confidence","act_type","dis_type","actual_disease","disease")
  plotdat=merge(sampdat_nn,dis_type, by="disease"); 
  plotdat = merge(plotdat, dat_classize[,c("dis","class_count")], by="dis")
  plotdat$class_fraction = plotdat$class_count/max(plotdat$class_count)
  tempfacts <- factor(plotdat$disease, levels=disease_order); newfactors = c(disease_order, unique(gsub("_[N,T]S","",plotdat[is.na(tempfacts),"dis"])))
  plotdat$disease <- factor(plotdat$disease, levels=newfactors)
  plotdat$confidence <- as.numeric(plotdat$confidence)
  plotdat$ycenter = 0; 
  
  plotdat$ensemble_prediction=gsub("_[N,T]S","",mostvoted_class[mostvoted_class$sample==smpl,"predicted"]); 
  plotdat$ensemble_prediction_type=gsub("[A-Z,-,a-z,MB-]*_","",mostvoted_class[mostvoted_class$sample==smpl,"predicted"]); 
  plotdat$ensemble_avg = mostvoted_class[mostvoted_class$sample==smpl,"avg_confidence"]
  plotdat$ensemble_machinecount = as.character(mostvoted_class[mostvoted_class$sample==smpl,"freq"])
  plotdat$ensemble_machinecount = factor(plotdat$ensemble_machinecount, levels=c(1,2,3,4,5))
  
  plotdat[is.na(plotdat$disease),"disease"] = gsub("_[N,T]S","",plotdat[is.na(plotdat$disease),"dis"]) #ESCA is not in the new disease order
  plotdat$Comparator = plotdat$dis_type
  plotdat[plotdat$dis_type=="NS","Comparator"] = "normal"; plotdat[plotdat$dis_type=="TS","Comparator"] = "tumour"
  plotdat$Comparator <- factor(plotdat$Comparator, levels=c("tumour","normal"))
  plotmedian <- stat_summary(fun.y="median",geom="point",size=4); plotrange = stat_summary(fun.data="mean_se",colour="grey",size=1,alpha=1,show.legend = TRUE) 
  plotmean <- stat_summary(fun.y="mean",geom="point",size=4);
  
  df_ensfacts = unique(plotdat[plotdat$dis_type==plotdat$ensemble_prediction_type & plotdat$dis==paste(plotdat$ensemble_prediction,"_",plotdat$ensemble_prediction_type,sep=""),c("ensemble_prediction","ensemble_avg","ensemble_machinecount","Source","Comparator")])
  
  biggest_class = dat_classize[dat_classize$class_count == max(dat_classize$class_count),]
  my_cSize_title <- paste(smpl," TCGA EnseN Confidence \n", "Bars show training fraction relative to biggest class size:", unique(biggest_class$dis), "(",biggest_class$class_count,")" , sep=" ")
   
  gp_tum = ggplot(plotdat[plotdat$dis_type=="TS",],aes(x=disease, y=confidence, colour=Source,fill=Source)) +
    plotrange + geom_point(size=3,alpha=0.3)+ plotmean + ylim(0,1) + guides(colour=FALSE, fill=FALSE ) + 
    facet_wrap(~actual_disease) + ggtitle(paste(my_cSize_title,"\n Tumour samples only",sep="")) + 
    facet_wrap(~Comparator, ncol=1)  + geom_point(data=plotdat[plotdat$dis_type==plotdat$ensemble_prediction_type,],pch=5, size=4, aes(x=ensemble_prediction, y=ensemble_avg), colour="black") + 
    plot_defaults + theme(axis.text.x = element_text(angle=45,hjust=1),plot.title = element_text(hjust = 0.5)) + 
    geom_text_repel(data=df_ensfacts, size=4, aes(x=ensemble_prediction, y=ensemble_avg, label=paste("Top-voted by \n ", ensemble_machinecount,"/", num_models," machines",sep="")), nudge_x=1.8) +
    geom_bar(data=unique(plotdat[plotdat$dis_type=="TS",c("disease","dis_type","class_fraction","Comparator","Source")]), aes(x=disease,y=class_fraction),fill="grey",colour="NA",alpha=0.25, size=2, stat="identity")  + facet_wrap(~Comparator,ncol=1,scales="fixed") 
  #+geom_label(data=df_ensfacts,size=4, aes(x=ensemble_prediction, y=ensemble_avg,label=ensemble_machinecount,fontface = 'bold')) 
  gp_split = ggplot(plotdat,aes(x=disease, y=confidence, colour=Source,fill=Source)) +plotrange + plotmean + geom_point(size=3,alpha=0.3) + plot_defaults+ ylim(0,1) + 
    guides(colour=FALSE, fill=FALSE ) + facet_wrap(~actual_disease) + ggtitle(my_cSize_title)  + facet_wrap(~Comparator, ncol=1)  + 
    geom_point(data=plotdat[plotdat$dis_type==plotdat$ensemble_prediction_type,],pch=5, size=4, aes(x=ensemble_prediction, y=ensemble_avg), colour="black") + 
    theme(axis.text.x = element_text(angle=45,hjust=1),plot.title = element_text(hjust = 0.5)) + geom_text_repel(data=df_ensfacts, size=4, aes(x=ensemble_prediction, y=ensemble_avg, label=paste("Top-voted by \n ", ensemble_machinecount,"/", num_models," machines",sep="")), nudge_x=1.8) +
    geom_bar(data=unique(plotdat[,c("disease","dis_type","class_fraction","Comparator","Source")]), aes(x=disease,y=class_fraction),colour="NA",fill="grey",alpha=0.25, size=2, stat="identity")  + facet_wrap(~Comparator,ncol=1,scales="fixed") 
  gp_split_range_cs = ggplot(plotdat,aes(x=disease, y=confidence, colour=Source,fill=Source)) +plotrange + plotmean + geom_point(size=3,alpha=0.3) + plot_defaults + 
    guides(colour=FALSE, fill=FALSE ) + facet_wrap(~actual_disease) + ggtitle(my_cSize_title) + facet_wrap(~Comparator, ncol=1, scales="free_y")  + 
    geom_point(data=plotdat[plotdat$dis_type==plotdat$ensemble_prediction_type,],pch=5, size=4, aes(x=ensemble_prediction, y=ensemble_avg), colour="black") + 
    theme(axis.text.x = element_text(angle=45,hjust=1),plot.title = element_text(hjust = 0.5))  + geom_text_repel(data=df_ensfacts, size=4, aes(x=ensemble_prediction, y=ensemble_avg, label=paste("Top-voted by \n ", ensemble_machinecount,"/", num_models," machines",sep="")), nudge_x=1.8) +
    geom_bar(data=unique(plotdat[,c("disease","dis_type","class_fraction","Comparator","Source")]), aes(x=disease,y=class_fraction),colour="NA",fill="grey",alpha=0.25, size=2, stat="identity")  + facet_wrap(~Comparator,ncol=1,scales="free_y") 
  gp_split_range = ggplot(plotdat,aes(x=disease, y=confidence, colour=Source,fill=Source)) +plotrange + plotmean + geom_point(size=3,alpha=0.3) + plot_defaults + 
    guides(colour=FALSE,fill=FALSE ) + facet_wrap(~actual_disease) + ggtitle(paste(smpl," TCGA EnseN Confidence ", sep=" ")) + facet_wrap(~Comparator, ncol=1, scales="free_y")  + 
    geom_point(data=plotdat[plotdat$dis_type==plotdat$ensemble_prediction_type,],pch=5, size=4, aes(x=ensemble_prediction, y=ensemble_avg), colour="black") + 
    theme(axis.text.x = element_text(angle=45,hjust=1),plot.title = element_text(hjust = 0.5))  + geom_text_repel(data=df_ensfacts, size=4, aes(x=ensemble_prediction, y=ensemble_avg, label=paste("Top-voted by \n ", ensemble_machinecount,"/", num_models," machines",sep="")), nudge_x=1.8)
  
  #Save plots to file (Make biopsy folder)
  ##Set up output directory and file names
  pdfpath=paste(plotdir,unique(sampdat$Biopsy),sep="/"); dir.create(pdfpath)
  cat("\n||| ", unique(sampdat$Biopsy), " ||| \n")
  ylabel = "Average confidence with SD spread \nIndividual values as smaller points" ; xlabel="Disease"
  tcpngfile=paste(pdfpath,"/", plotprefix, "_",smpl, "_tumour.png",sep="")
  if(0) {#smpl_tc > 0){
    ggsave(filename=tcpngfile, plot=gp_tum + labs(x=xlabel, y=ylabel) + geom_hline(aes(yintercept=smpl_tc), linetype="dashed"), width=14, height=9 )
  }else{
    ggsave(filename=tcpngfile, plot=gp_tum + labs(x=xlabel, y=ylabel), width=14, height=9 )
  }
  
  tcnfile=paste(pdfpath,"/", plotprefix,"_",smpl, "_tumour_normal.png",sep="")
  ggsave(filename=tcnfile, plot=gp_split+ labs(x=xlabel, y=ylabel), width=14, height=10 )
  
  tcpngfile=paste(pdfpath,"/", plotprefix, "_",smpl, "_tumour_normal_zoom.png",sep="")
  ggsave(filename=tcpngfile, plot=gp_split_range_cs + labs(x=xlabel, y=ylabel), width=14, height=9 )
  
  tcnfile=paste(pdfpath,"/", plotprefix,"_",smpl, "_tumour_normal_zoom_noclassSize.png",sep="")
  ggsave(filename=tcnfile, plot=gp_split_range + labs(x=xlabel, y=ylabel), width=14, height=10 )
  
  cat("~~~~ FIGURES generated at ",pdfpath,"\n")
   
  plot_t[[i]] = gp_tum
  plot_all_nt[[i]] = gp_split
  plot_all_ntscale[[i]]=gp_split_range
  i = i+1

}

