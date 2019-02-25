##Get required libs
#.libPaths("/home/jgrewal/software/mac_home/R_libs/")
library(stringr);library(reshape2);library(plyr); library(ggplot2); library(gtable); library(gdata)
library(RColorBrewer)
plot_defaults=theme_bw(base_size = 18)+ theme(axis.text.x = element_text(angle=90,hjust=1)) 

#Get required reference dataframes
diseasedat=read.table("/home/jgrewal/mygits/cancerscope/cancerscope/resources/custom_ped_pog_diseasetypes_v8.txt",check.names=FALSE,sep="\t",fill=TRUE,header=TRUE,row.names=NULL,stringsAsFactors = FALSE)
dis_type=unique(diseasedat[,c("Source","Code Name")]); colnames(dis_type) = c("Source","disease")
order_source=c("Central Nervous System","Endocrine","Hematologic","Head and Neck", "Skin","Thoracic","Breast","Urologic","Gastrointestinal","Soft Tissue","Gynecologic","Non specific","Neuroblastoma")
dis_type$Source=factor(dis_type$Source,levels = order_source)
disease_order=c("MB-Adult","GBM","TFRI_GBM_NCL","LGG","THCA","FL","DLBC","NCI_GPH_DLBCL","DLBC_BM","THYM","LAML","UVM","SKCM","MESO","LUAD","BRCA","PCPG","ACC","KIRC","KIRP","KICH","LIHC","PAAD","STAD","COADREAD","ESCA_EAC","CHOL","TGCT","BLCA","PRAD","SARC","OV","UCS","UCEC","CESC_CAD","HNSC","LUSC","ESCA_SCC","CESC_SCC")

##Make a dis_type palette
getPalette <- colorRampPalette(brewer.pal(9, "Set3"))
myColors = getPalette(length(order_source))
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
myColors = gg_color_hue(length(order_source))
names(myColors) <- levels(dis_type$Source)
colScale <- scale_colour_manual(name = "Source",values = myColors)
fillScale <- scale_fill_manual(name="Source",values=myColors)

## Get predicted 
get_predicted <- function(path_with_NNoutputs, return_models=FALSE, look_recursive=FALSE){
  dat_with_sample_model_topconf_run = data.frame()
  for(p in path_with_NNoutputs){
    p1 = getdatfiles(p, in_cols=c("predicted", "confidence", "sample"), look_recursive=look_recursive)
    dat_with_sample_model_topconf_run = rbind(dat_with_sample_model_topconf_run, p1)
  }
  return(get_most_voted(dat_with_sample_model_topconf_run, return_models=return_models))
}

#Get most voted class for each sample. In case 2 classes get most votes, select the class with higher average confidence in the models that picked it best
get_most_voted <- function(dat_with_sample_model_topconf, return_models=FALSE){
  #Input must have sample, predicted (class), confidence (confidence of highest call by each model), file (representing model)
  bestcall=data.frame()
  for(sample in unique(dat_with_sample_model_topconf$sample)){
      sample_rows = dat_with_sample_model_topconf[(dat_with_sample_model_topconf$sample==sample),]
      sample_rows_ref = dat_with_sample_model_topconf[(dat_with_sample_model_topconf$sample==sample),]
      meanforbestclass <- ddply(sample_rows[,c("sample","predicted","confidence")],c("sample","predicted"), summarise, avg_confidence=mean(confidence), freq=length(predicted))
      if(max(meanforbestclass$avg_confidence)>1){
        sample_rows=meanforbestclass[meanforbestclass$avg_confidence == max(meanforbestclass$avg_confidence),]
      }else{
        sample_rows=meanforbestclass[meanforbestclass$freq==max(meanforbestclass$freq),]
        sample_rows = sample_rows[sample_rows$avg_confidence == max(sample_rows$avg_confidence),]
        if(return_models){
          sample_rows_ref$file = gsub("TESTPRED_","",sample_rows_ref$file); 
          sample_rows_ref$file = gsub("_.*","",sample_rows_ref$file); 
          sample_rows$models=do.call(paste,c(as.list(sort(as.character(sample_rows_ref[sample_rows_ref$predicted==sample_rows$predicted,"file"]))),sep=";"))
        }
      }
      bestcall=rbind(bestcall, sample_rows)
  }
  return(bestcall)
}

get_most_voted_weighted <- function(dat_with_sample_model_topconf, return_models=FALSE){
  #Input must have sample, predicted (class), confidence (confidence of highest call by each model), file (representing model)
  dat_with_sample_model_topconf$confidence_weighted = dat_with_sample_model_topconf$confidence
  model_weighted_df = data.frame(modelname=unique(dat_with_sample_model_topconf$model),modelweight=1)
  model_weighted_df[model_weighted_df$modelname=="smotenone17k","modelweight"]=0.827
  model_weighted_df[model_weighted_df$modelname=="rm500dropout","modelweight"]=0.9275
  model_weighted_df[model_weighted_df$modelname=="rm500","modelweight"]=0.846
  model_weighted_df[model_weighted_df$modelname=="none17kdropout","modelweight"]=0.6677
  model_weighted_df[model_weighted_df$modelname=="none17k","modelweight"]=0.8035
  for(i in unique(dat_with_sample_model_topconf$model)){
    dat_with_sample_model_topconf[dat_with_sample_model_topconf$model==i, "confidence_weighted"] =dat_with_sample_model_topconf[dat_with_sample_model_topconf$model==i, "confidence_weighted"] * model_weighted_df[model_weighted_df$modelname==i,"modelweight"]
  }
  bestcall_confidence = dat_with_sample_model_topconf[,c("predicted","confidence","file","pog","biopsy","sample")]
  bestcall_confidence_weighted = dat_with_sample_model_topconf[,c("predicted","confidence_weighted","file","pog","biopsy","sample")]
  colnames(bestcall_confidence_weighted) = c("predicted","confidence","file","pog","biopsy","sample")
  bestcall_confidence = get_most_voted(bestcall_confidence, return_models)
  bestcall_confidence_weighted = get_most_voted(bestcall_confidence_weighted, return_models)
  return(bestcall_confidence_weighted)
}

get_most_voted_iterativemax <- function(dat_with_sample_model_confidences, return_models=FALSE){
  #Input must have sample, model, and each of the predicted classes as output
  bestcall_confidence = data.frame()
  for(s in unique(dat_with_sample_model_confidences$sample)){
    dat_with_sample_models = dat_with_sample_model_confidences[dat_with_sample_model_confidences$sample==s,]
    sample_name = unique(dat_with_sample_models$pog)
    tdat_num = dat_with_sample_models[,sapply(dat_with_sample_models, is.numeric)]
    tdat_num$confidence <- NULL
    tdat_num$model = dat_with_sample_models$model
    bestcall_sample = get_most_voted_fromALLclasses(tdat_num)
    bestcall_sample$sample = s
    bestcall_sample$sample_name = sample_name
    bestcall_confidence = rbind(bestcall_confidence, bestcall_sample)
  }
  return(bestcall_confidence)
}

get_most_voted_fromALLclasses <- function(dat_with_cols_as_classes, num_loops=0){
  models = dat_with_cols_as_classes[,ncol(dat_with_cols_as_classes)] #last column is the models
  max_per_row = colnames(dat_with_cols_as_classes)[apply(dat_with_cols_as_classes[,-ncol(dat_with_cols_as_classes)], 1, which.max)]
  max_conf_per_row = unname(apply(dat_with_cols_as_classes[,-ncol(dat_with_cols_as_classes)], 1, max))
  how_many_topvoterMachines = max(count(max_per_row)$freq)
  which_topvoter = as.character(count(max_per_row)[count(max_per_row)$freq == how_many_topvoterMachines,"x"])
  minimum_votes= as.integer(dim(dat_with_cols_as_classes)[1]/2)
  num_loops= num_loops + 1
  
  if(how_many_topvoterMachines < minimum_votes & dim(dat_with_cols_as_classes)[2] > 2){
    #If less than 2 of machines have an identical clear top vote 
    #Remove the class with the highest confidence across all machines
    topvoter_row = dat_with_cols_as_classes[which.max(max_conf_per_row),]
    topvoter_class = max_per_row[which.max(max_conf_per_row)]
    new_confdat = dat_with_cols_as_classes[,!(colnames(dat_with_cols_as_classes) ==topvoter_class)]
    get_most_voted_fromALLclasses(new_confdat, num_loops)
  }else{
    if(length(which_topvoter)==2){
      #if there is a tie, i.e. 2 machines voted for m and 2 voted for p, fix topvoter based on max average confidence across all models
      #Approach #1, if more models voted higher for a particular class, choose that; if same number of models voting for each, then go with the avg higher one in the model subset that is higher than the other class
      eachclass_preds = dat_with_cols_as_classes[,which_topvoter]
      tempdat = data.frame()
      for(i in unique(colnames(eachclass_preds))){
        max_vote_i = mean(eachclass_preds[eachclass_preds[,i] > eachclass_preds[,colnames(eachclass_preds) != i],i])
        num_higher = length((eachclass_preds[eachclass_preds[,i] > eachclass_preds[,colnames(eachclass_preds) != i],i]))
        tempdat = rbind(tempdat, data.frame(class=i, mean_conf=max_vote_i, model_count=num_higher))
      }
      if(max(tempdat$model_count) > 2){which_topvoter = tempdat[tempdat$model_count==max(tempdat$model_count),"class"]}else{
        which_topvoter = tempdat[tempdat$mean_conf==max(tempdat$mean_conf),"class"]
      }
      #Approach 2, Get mean confidence across ALL MODELS
      ##mean_confidence_across_all_models = as.data.frame(apply(dat_with_cols_as_classes[,which_topvoter] , 2, mean))
      ##which_topvoter=rownames(mean_confidence_across_all_models)[(apply(mean_confidence_across_all_models,2,which.max))]
    }
    #Otherwise, if 3 or more machines voted for the same thing, go with that
    which_topvoter_models = models[which(max_per_row == which_topvoter)]
    avg_conf = mean((dat_with_cols_as_classes[which(max_per_row == which_topvoter),which_topvoter]))
    highest_set = data.frame(predicted = which_topvoter, freq = how_many_topvoterMachines, avg_confidence=avg_conf, models=paste(sort(models[which(max_per_row == which_topvoter)]), collapse= ";"), iter_level=num_loops)
    return(highest_set)
  }
}

rescale_predmodeldat <- function(pdat){
  #pdat has one column which we group by, model, and then rescale column confidence
  rdat = data.frame()
  for(i in unique(pdat$model)){
    datemp = pdat[pdat$model==i,]
    datemp$confidence_scaled = rescale(datemp$confidence,range(0,1))
    rdat = rbind(rdat,datemp)
  }
  return(rdat)
}
#Define confusion matrix function
plot.confusion_agg<- function(conf)
{colnames(conf) <- c("True","Predicted","method")
conf$pred_status=1; 
conf2 <- aggregate(pred_status~True+Predicted+method,data=conf,FUN=sum)
melted <- reshape2::melt(conf2, c("True", "Predicted","method"))
melted <- join(melted,count(conf,"True"))
p <-ggplot(data=melted, aes(x=Predicted, y=True,fill=method))
p <- p + geom_point(shape=22,alpha=0.5, aes(size=(value/freq))) + theme_bw() #,position=position_jitter(width=1,height=0))
p <- p + theme(axis.text.x = element_text(angle=90,hjust=1)) 
p <- p + guides(fill=guide_legend(title="Total cases in class",override.aes=list(size=5)), size=guide_legend(title="Fraction of True class predicted"))
p <- p + geom_point(aes(x=True,y=True,size=(value/n)), size=8, fill=NA,shape=21)+ scale_fill_manual(values=c("red1", "steelblue1"))
p
}


plot.confusion<- function(conf){
  colnames(conf) <- c("True","Predicted")
  conf$pred_status=1; 
  conf2 <- aggregate(pred_status~True+Predicted,data=conf,FUN=sum)
  melted <- reshape2::melt(conf2, c("True", "Predicted"))
  melted <- join(melted,count(conf,"True")); colnames(melted) = c("True","Predicted","variable","value","n")
  p <-ggplot(data=melted, aes(x=Predicted, y=True))
  p <- p + geom_point(shape=22,aes(fill=n,size=(value/n))) + theme_bw(base_size = 18)
  p <- p + theme(axis.text.x = element_text(angle=90,hjust=1)) 
  p <- p + guides(fill=guide_legend(title="Total cases in class",override.aes=list(size=5)), size=guide_legend(title="Fraction of True class predicted"))+ scale_fill_gradientn(colours=topo.colors(5))
  p <- p + geom_point(aes(x=True,y=True,size=(value/n)), size=8, fill=NA,shape=21)
  p
}

plot.ordered.confusion_agg<- function(conf,ordering_dict=dis_type)
{colnames(conf) <- c("True","Predicted","method")
conf$pred_status=1; 
conf2 <- aggregate(pred_status~True+Predicted+method,data=conf,FUN=sum)
melted <- reshape2::melt(conf2, c("True", "Predicted","method"))
melted <- join(melted,plyr::count(conf,"True"))
melted$True_class = gsub("_[N,T]S$","",melted$True);
melted$Predicted_class = gsub("_[N,T]S$","",melted$Predicted);
if(!(is.na(ordering_dict))){
dat3=merge(melted,ordering_dict, by.x="Predicted_class", by.y="disease"); dat3$Source_predicted = dat3$Source
dat3=merge(dat3,ordering_dict, by.x="True_class", by.y="disease"); dat3$Source_true = dat3$Source.y

p <-ggplot(data=dat3, aes(x=Predicted, y=True))
p <- p + geom_point(pch=10,aes(x=True, y=True,colour=Source_true,size=freq),alpha=0.50)  + theme_bw(base_size = 18) #,position=position_jitter(width=1,height=0))
p <- p + geom_point(aes(colour=Source_predicted, size=value, alpha=max(0.1,value/freq)))
p <- p + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5,pch=16)), size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
p <- p + colScale + facet_grid(method~.)
p
}else{
  
  p <-ggplot(data=melted, aes(x=Predicted, y=True))
  p <- p + geom_point(pch=10,aes(x=True, y=True,colour=method,size=freq),alpha=0.50)  + theme_bw(base_size = 18) #,position=position_jitter(width=1,height=0))
  p <- p + geom_point(aes(size=value, alpha=max(0.1,value/freq)))
  p <- p + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
  p <- p + colScale #+ facet_grid(method~.)
  p
}

}

plot.ordered.confusion<- function(conf, ordering_dict=dis_type){
  colnames(conf) <- c("True","Predicted")
  conf$pred_status=1; 
  conf2 <- aggregate(pred_status~True+Predicted,data=conf,FUN=sum)
  melted <- reshape2::melt(conf2, c("True", "Predicted"))
  melted <- join(melted,plyr::count(conf,"True")); colnames(melted) = c("True","Predicted","variable","value","n")
  melted$True_class = gsub("_[N,T]S$","",melted$True);
  melted$Predicted_class = gsub("_[N,T]S$","",melted$Predicted);
  if(!(is.null(ordering_dict))){
  dat3=merge(melted,ordering_dict, by.x="Predicted_class", by.y="disease"); dat3$Source_predicted = dat3$Source
  dat3=merge(dat3,ordering_dict, by.x="True_class", by.y="disease"); dat3$Source_true = dat3$Source.y
  
  #ggplot(data=dat3, aes(x=Predicted, y=True)) + geom_point(aes(colour=Source_predicted, size=value))+ geom_point(aes(x=True, y=True,colour=Source_true,size=n),alpha=0.20)  + theme_bw(base_size = 18) + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5)), size=guide_legend(title="Fraction of True class predicted"))
  p <- ggplot(data=dat3, aes(x=Predicted, y=True)) 
  p <- p + geom_point(pch=10,aes(x=True, y=True,colour=Source_true,size=n),alpha=0.50)  + theme_bw(base_size = 18) 
  p <- p + geom_point(aes(colour=Source_predicted, size=value, alpha=max(0.1,value/n)))
  p <- p + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5,pch=16)), size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
  p + colScale 
  }
  else{
    #ggplot(data=dat3, aes(x=Predicted, y=True)) + geom_point(aes(colour=Source_predicted, size=value))+ geom_point(aes(x=True, y=True,colour=Source_true,size=n),alpha=0.20)  + theme_bw(base_size = 18) + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5)), size=guide_legend(title="Fraction of True class predicted"))
    p <- ggplot(data=melted, aes(x=Predicted, y=True)) 
    p <- p + geom_point(pch=10,aes(x=True, y=True,size=n),alpha=0.50)  + theme_bw(base_size = 18) 
    p <- p + geom_point(aes(size=value, alpha=max(0.1,value/n)))
    p <- p + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides( size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
    p + colScale 
  }
}
plot.ordered.normtum.agg.confusion<- function(conf, ordering_dict=dis_type,splitsets=TRUE){
  colnames(conf) <- c("True","Predicted","model_type")
  plotdat=data.frame()
  for(i in unique(conf$model_type)){
    confsub=conf[conf$model_type==i,c("True","Predicted")]
    confsub$pred_status=1; 
    conf2 <- aggregate(pred_status~True+Predicted,data=confsub,FUN=sum)
    melted <- melt(conf2, c("True", "Predicted"))
    melted <- join(melted,count(conf,"True")); colnames(melted) = c("True","Predicted","variable","value","n")
    melted$True_class = gsub("_[N,T]S$","",melted$True); melted$True_type=gsub("[A-Z,-,a-z,MB-]*_","",melted$True)
    melted$Predicted_class = gsub("_[N,T]S$","",melted$Predicted);melted$Predicted_type=gsub("[A-Z,-,a-z,MB-]*_","",melted$Predicted)
    dat3=merge(melted,ordering_dict, by.x="Predicted_class", by.y="disease"); dat3$Source_predicted = dat3$Source
    dat3=merge(dat3,ordering_dict, by.x="True_class", by.y="disease"); dat3$Source_true = dat3$Source.y
    
    dat3$model_type=i
    plotdat=rbind(plotdat,dat3)
  }
  #ggplot(data=dat3, aes(x=Predicted, y=True)) + geom_point(aes(colour=Source_predicted, size=value))+ geom_point(aes(x=True, y=True,colour=Source_true,size=n),alpha=0.20)  + theme_bw(base_size = 18) + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5)), size=guide_legend(title="Fraction of True class predicted"))
  
  p1 <- ggplot(data=plotdat[plotdat$Predicted_type=="TS",], aes(x=Predicted, y=True)) + geom_point(pch=10,aes(x=True, y=True,colour=Source_true,size=n),alpha=0.50)
  p1 <- p1 + geom_point(aes(colour=Source_predicted, size=value, alpha=max(0.1,value/n)))+ facet_grid(~model_type)
  p1<- p1   + theme_bw(base_size = 18) + colScale
  p1 <- p1 + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5,pch=16)), size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
  #p1
  p2 <- ggplot(data=plotdat[plotdat$Predicted_type=="NS",], aes(x=Predicted_class, y=True)) + geom_point(pch=10,aes(x=True_class, y=True,colour=Source_true,size=n),alpha=0.50)
  p2 <- p2 + geom_point(aes(colour=Source_predicted, size=value, alpha=max(0.1,value/n)))+ facet_grid(~model_type)
  p2 <- p2   + theme_bw(base_size = 18) + ggtitle("Tumour samples misclassified as Normals") + colScale
  p2 <- p2 + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5,pch=16)), size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
  #
  p3 <- ggplot(data=plotdat[plotdat$Predicted_type=="NS" & plotdat$True_type!="NS",], aes(x=Predicted, y=True)) + geom_point(pch=10,aes(x=True, y=True,colour=Source_true,size=n),alpha=0.50)
  p3 <- p3 + geom_point(aes(colour=Source_predicted, size=value, alpha=max(0.1,value/n)))+ facet_grid(~model_type)
  p3 <- p3   + theme_bw(base_size = 18) + ggtitle("Tumour samples misclassified as Normals") + colScale
  p3 <- p3 + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5,pch=16)), size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
  
  p4 <- ggplot(data=plotdat[plotdat$Predicted_type=="TS" & plotdat$True_type!="TS",], aes(x=Predicted, y=True)) + geom_point(pch=10,aes(x=True, y=True,colour=Source_true,size=n),alpha=0.50)
  p4 <- p4 + geom_point(aes(colour=Source_predicted, size=value, alpha=max(0.1,value/n)))+ facet_grid(~model_type)
  p4 <- p4   + theme_bw(base_size = 18) + ggtitle("Normal samples misclassified as Tumours")
  p4 <- p4 + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5,pch=16)), size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
  
  p5 <- ggplot(data=plotdat, aes(x=Predicted, y=True)) + geom_point(pch=10,aes(x=True, y=True,colour=Source_true,size=n),alpha=0.50)
  p5 <- p5 + geom_point(aes(colour=Source_predicted, size=value, alpha=max(0.1,value/n)))+ facet_grid(model_type~.,scales="free")
  p5<- p5   + theme_bw(base_size = 18) + colScale
  p5 <- p5 + scale_size(range=c(4,10)) + theme(axis.text.x = element_text(angle=90,hjust=1)) + scale_alpha(guide='none') + guides(colour=guide_legend(title="Tissue type",override.aes=list(size=5,pch=16)), size=guide_legend(title="Number of Samples Predicted",override.aes=list(pch=21)))
  
  if(splitsets==FALSE){
    return(list(p1,p2,p3,p4, p1 + facet_grid(~model_type,scales = "free"), p5))
  }else{
    g1<-ggplotGrob(p1+ theme(panel.margin.x=unit(8, "lines"),panel.margin.y=unit(6,"lines"),axis.text.y=element_text(hjust=0.5))); 
    axis <- gtable_filter(g1, "axis-l")[["grobs"]][[1]][["children"]][["axis"]][,1]
    g1[["grobs"]][[4]][["children"]][["axis"]] <- NULL
    panels <- subset(g1$layout, name == "panel")
    g1 <- gtable_add_grob(g1, grobs=axis, t = unique(panels$t), l=tail(panels$l, -1)-1)
    #grid.newpage(); grid.draw(g1)
    
    g2<-ggplotGrob(p2+ theme(panel.margin.x=unit(8, "lines"),panel.margin.y=unit(6,"lines"),axis.text.y=element_text(hjust=0.5))); 
    axis <- gtable_filter(g2, "axis-l")[["grobs"]][[1]][["children"]][["axis"]][,1]
    g2[["grobs"]][[4]][["children"]][["axis"]] <- NULL
    panels <- subset(g2$layout, name == "panel")
    g2 <- gtable_add_grob(g2, grobs=axis, t = unique(panels$t), l=tail(panels$l, -1)-1)
    return(list(p1,p2,g1,g2,p3,p4))
  }
}

get.stats <- function(conf){
  colnames(conf) <- c("True","Predicted")
  conf$True = as.character(conf$True); conf$Predicted=as.character(conf$Predicted)
  cm = as.matrix(table(Actual=conf$True, Predicted=conf$Predicted))
  missing_trues = setdiff(colnames(cm),rownames(cm))
  zeromat = matrix(nrow=length(missing_trues),ncol= ncol(cm)); rownames(zeromat)=missing_trues; colnames(zeromat)=colnames(cm); cm = rbind(cm,zeromat)
  missing_falses = setdiff(rownames(cm),colnames(cm))
  zeromat = matrix(ncol=length(missing_falses),nrow= nrow(cm)); rownames(zeromat)=rownames(cm); colnames(zeromat)=missing_falses; cm = cbind(cm,zeromat)
  cm=cm[order(rownames(cm)),order(colnames(cm))]; cm[is.na(cm)] = 0
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  diag = diag(cm) # number of correctly classified instances per class 
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes
  accur = sum(diag)/n
  accuracy = diag/rowsums
  precision = diag / colsums 
  recall = diag / rowsums 
  f1 = 2 * precision * recall / (precision + recall) 
  summary_stat = data.frame(accur,accuracy,precision, recall, f1)[unique(conf$True),] 
  return(replace(summary_stat, is.na(summary_stat),0))
}

measurePrecisionRecall <- function(predict, actual_labels){
  precision <- sum(predict & actual_labels) / sum(predict)
  recall <- sum(predict & actual_labels) / sum(actual_labels)
  fmeasure <- 2 * precision * recall / (precision + recall)
  
  cat('precision:  ')
  cat(precision * 100)
  cat('%')
  cat('\n')
  
  cat('recall:     ')
  cat(recall * 100)
  cat('%')
  cat('\n')
  
  cat('f-measure:  ')
  cat(fmeasure * 100)
  cat('%')
  cat('\n')
}

getdatfiles<- function(path, pattern="ordered", header_status=FALSE, look_recursive=TRUE,in_cols= c("nn_pred","nn_score","analysis_used","sample")){
  #Needs stringr library
  infiles=list.files(path=path, pattern=pattern, recursive=look_recursive,full.names=TRUE)
  if(is.null(in_cols)){header_status=TRUE}
  dat_full=data.frame()
  for(i in infiles){
    dat<-read.table(i,sep="\t", header=header_status, stringsAsFactors = FALSE);
    if(!(is.null(in_cols))){colnames(dat) <- in_cols;}
    dat$file=i; 
    dat$file = str_match(dat$file, "TESTPRED_.*/"); 
    dat$pog = str_match(dat$sample,"POG[0-9][0-9][0-9]")
    dat_full=rbind(dat,dat_full)
  }
  return(dat_full)
}

