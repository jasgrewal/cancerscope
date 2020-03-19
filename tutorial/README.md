# Tutorial

This tutorial will go through the use of cancerscope to predict the cancer type from a) an input file, or b) from pre-loaded RNA-Seq data.  
We will be using some example RNA-Seq data from TCGA. You can download the data file which has been pre-collated for you [here](combined_tcga_fpkm.txt).  


## (Optional) Collating example data yourself  
You can also prepare this data yourself. Download the data using the [gdc-rnaseq-tool](https://github.com/cpreid2/gdc-rnaseq-tool) and this [TCGA query](https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22%3E%3D%22%2C%22content%22%3A%7B%22field%22%3A%22cases.diagnoses.age_at_diagnosis%22%2C%22value%22%3A%5B6574%5D%7D%7D%2C%7B%22op%22%3A%22%3C%3D%22%2C%22content%22%3A%7B%22field%22%3A%22cases.diagnoses.age_at_diagnosis%22%2C%22value%22%3A%5B7304%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-HNSC%22%2C%22TCGA-LGG%22%2C%22TCGA-LIHC%22%2C%22TCGA-PCPG%22%2C%22TCGA-SKCM%22%2C%22TCGA-TGCT%22%2C%22TCGA-THCA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.access%22%2C%22value%22%3A%5B%22open%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.analysis.workflow_type%22%2C%22value%22%3A%5B%22HTSeq%20-%20FPKM%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22TXT%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Gene%20Expression%20Quantification%22%5D%7D%7D%5D%7D&searchTableTab=files)  

The example data used is then sourced as follows:  
`https://gdc.cancer.gov/access-data/gdc-data-transfer-tool`
`gdc-client download -m gdc_manifest_age22.txt`  

## Getting started
Please install cancerscope, and download all files in this directory. Particularly, make sure you have downloaded the file **combined_tcga_fpkm.txt**  

### Package import and setup   
Start by importing the package into your python instance.  
`>>> import cancerscope as cs`  

If this is your first time importing **cancerscope**, You will be greeted with the following output:   
 
```python   
Thankyou for using cancerscope. The initial run requires download of dependent model files. Proceeding with download now...
	Models will be downloaded to /PATH_TO_PTYHON_LIB/lib/python2.7/site-packages/cancerscope/data/

Downloading model files for v1_rm500dropout 
	Data Downloaded at: /PATH_TO_PYTHON_LIB/lib/python2.7/site-packages/cancerscope/data/
Downloading model files for v1_none17kdropout 
	Data Downloaded at: /PATH_TO_PYTHON_LIB/lib/python2.7/site-packages/cancerscope/data/
Downloading model files for v1_rm500 
	Data Downloaded at: /PATH_TO_PYTHON_LIB/lib/python2.7/site-packages/cancerscope/data/
Downloading model files for v1_smotenone17k 
	Data Downloaded at: /PATH_TO_PYTHON_LIB/lib/python2.7/site-packages/cancerscope/data/
Downloading model files for v1_none17k 
	Data Downloaded at: /PATH_TO_PYTHON_LIB/lib/python2.7/site-packages/cancerscope/data/
```

### Fitting the models  

Set up the predictor...   
`>>> scope_obj=cs.scope()`     

### Obtaining predictions  

Process data from file...   
`>>> pfromfile = scope_obj.get_predictions_from_file("/PATH/TO/combined_tcga_fpkm.txt")`   

You should see the following output:   

```python    
Your genenames are of format ENSG.version. Truncating to ENSG ID only

Read in sample file /PATH/TO/combined_tcga_fpkm.txt, 
	Data shape (26, 60483)
	Number of samples 26
	Number of genes in input 60483, with gene code ENSEMBL
...Only 17645 of input features mapped to expected number of features. Setting the rest to 0.0...Normalization function being applied: rastminmax
norm result shape (26, 17688)

...Only 17645 of input features mapped to expected number of features. Setting the rest to 0.0...Normalization function being applied: none
norm result shape (26, 17688)

...Only 17645 of input features mapped to expected number of features. Setting the rest to 0.0...Normalization function being applied: rastminmax
norm result shape (26, 17688)

...Only 17645 of input features mapped to expected number of features. Setting the rest to 0.0...Normalization function being applied: none
norm result shape (26, 17688)

...Only 17645 of input features mapped to expected number of features. Setting the rest to 0.0...Normalization function being applied: none
norm result shape (26, 17688)
```     

In some cases, the provided gene names will not map over exactly to the list of gene names used by SCOPE (thankyou 100 different ways of naming a gene). The count of matching genes will be output to you for every model.    

**The output object will look like this:**  

```python
>>> pfromfile
    sample_ix    label      pred  freq                                             models  rank_pred             sample_name
0           0  SKCM_TS  0.977178     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_SKCM_65312630
1           1  THCA_TS  0.999639     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_THCA_b2016510
2           2  THCA_NS  0.909720     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_THCA_ffb8427a
3           3  TGCT_TS  0.853214     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_TGCT_78264e6b
4           4  THCA_TS  0.943202     3         v1_rm500dropout,v1_none17kdropout,v1_rm500          1      TCGA_THCA_4869e2a4
5           4  LUAD_TS  0.570298     2                         v1_smotenone17k,v1_none17k          2      TCGA_THCA_4869e2a4
6           5  THCA_TS  0.900357     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_THCA_73451252
7           6  HNSC_TS  0.993321     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_HNSC_26019321
8           7  SKCM_TS  0.915901     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_SKCM_22632bc1
9           8  PCPG_TS  0.998648     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_PCPG_cf680d44
10          9  THCA_TS  0.987237     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_THCA_34826584
11         10  THCA_TS  0.994749     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_THCA_5fedc450
12         11  PCPG_TS  0.999465     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_PCPG_4f16b358
13         12  THCA_TS  0.992660     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_THCA_4640600f
14         13  TGCT_TS  0.821080     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_TGCT_d8ad327f
15         14  THCA_NS  0.915354     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_THCA_e32c7fe0
16         15  LIHC_TS  0.996797     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_LIHC_abe89868
17         16  PCPG_TS  0.987895     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_PCPG_1182a295
18         17  TGCT_TS  0.947827     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_TGCT_a51c7a87
19         18  SKCM_TS  0.863788     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_SKCM_bcc52bb7
20         19  THCA_TS  0.856415     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1  TCGA_THCA_NHL_a7229653
21         20  TGCT_TS  0.824165     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_TGCT_af5c9e80
22         21  SKCM_TS  0.935106     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1      TCGA_SKCM_074f955e
23         22   LGG_TS  0.998421     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1       TCGA_LGG_9cd81de0
24         23  THCA_TS  0.975626     3         v1_rm500dropout,v1_none17kdropout,v1_rm500          1      TCGA_THCA_7429a1e0
25         23  LUAD_TS  0.691465     2                         v1_smotenone17k,v1_none17k          2      TCGA_THCA_7429a1e0
26         24  LUAD_TS  0.765275     3       v1_none17kdropout,v1_smotenone17k,v1_none17k          1      TCGA_THCA_94cda1d7
27         24  THCA_TS  0.994200     2                           v1_rm500dropout,v1_rm500          2      TCGA_THCA_94cda1d7
28         25   LGG_TS  0.998769     5  v1_rm500dropout,v1_none17kdropout,v1_rm500,v1_...          1       TCGA_LGG_cfd39475
```

### Interpreting the output   
The sample index is the order the samples were read in. The `label` is the predicted class. `\_[N,T]S` indicates a healthy normal (N) or a tumour type (T). The tumor codes follow TCGA naming guidelines. Please refer to the publication for detailed names.   

The `pred` is the prediction confidence for the `label`, from the models listed in the column `models`. The `freq` is the count of models whose top-prediction was `label`. The `freq` column matches the number of columns listed in `models`.   

The `rank_pred` ranks the predictions per sample. In cases where the voting was unanimous, you see that the sample has a single rank (`rank_pred` == 1), and `freq` == 5.  

In some instances in the example above, the prediction is split between two classes. The user can pick the highest ranked prediction for each case (`rank_pred` == 1), or use their better judgement to infer the results.  

#### Why don't I just use the top-ranked prediction?   
The intrepretation of confidence can be nuanced in some cases.   

	- For example, consider the sample "TCGA_THCA_94cda1d7" (sample_ix = 24). Here, 3 models predicted that the cancertype was LUAD (lung adenocarcinoma), with an average confidence of 0.765155. As a higher number of models predicted this, the rank of prediction is 1 (rank_pred = 1).  
	
	- However, 2 models also predicted the THCA (thyroid carcinoma) tumor type, which is correct. The rank of prediction in this case is 2, because a fewer number of models predicted this (even though the average confidence was much higher, at 0.994180). The two results are provided to the user to facilitate a judgement call.     
	
	- In contrast, see the case "TCGA_THCA_7429a1e0". Here, the average confidence *and* the number of models contributing to the call of 'THCA_TS' is higher, than those calling the case 'LUAD_TS'. In such a scenario, the user can simply choose to discard the 2nd ranked prediction, and go only with the 1st one, since both measures of certainty are higher in rank_pred == 1.  

### Plotting the output, or saving the output as txt file  
If you provide the prediction call with an output directory, it will generate the following:  
1. A txt file listing the dataframe returned above.  
2. A txt file listing the prediction confidence from each model, across all 66 classes, for all samples.  
3. An individual svg file labelled with the sample name, for each sample.  

`>>> pfromfile = scope_obj.get_predictions_from_file("/PATH/TO/combined_tcga_fpkm.txt", outdir="/PATH/TO/DESIRED/OUTPUT/FOLDER/")`  

### Only generating predictions from a subset of models  
If you like to play favorites, or generally dislike the sound of a particular model, you can tell SCOPE which models to consider when assessing an input.  

`>>> pfromfile = scope_obj.get_predictions_from_file("/PATH/TO/combined_tcga_fpkm.txt", modelnames=[LIST OF MODELS TO KEEP])`

The model names for Version 1.0 of **cancerscope** are as follows:  
v1_rm500  
v1_rm500dropout  
v1_smotenone17k  
v1_none17k  
v1_none17kdropout  

Please refer to the supplementary data in the publication for details on each model.  
 
### FAQs  
What is the GENE\_IDENTIFIER?
The Gene Identifier has to be uniform across the list (don't provide a mix of ENSG, HUGO, and ENTREZ ids, for example). If passing in a file, the column name for the first column ('GENE\_IDENTIFIER') should indicate the type of genenames. Options are as follows:  

|SCOPE|ENTREZ|HUGO|GENENAME|ENSEMBL|HGNC|GSC1|GSC2|HUGO_ENSG|SCOPE_ENSG|   
|---|---|---|---|---|---|---|---|---|---|   
|A1BG|1|A1BG|A1BG|ENSG00000121410|5|A1BG\|1_calculated|merged_AIBG\|1\_calculated|A1BG_ENSG00000121410|A1BG_ENSG00000121410|    
|WBSCR17|64409|GALNT17|GALNT17|ENSG00000185274|16347|WBSCR17\|64409_calculated|merged_WBSCR17\|64409_calculated|GALNT17\_ENSG00000185274|WBSCR17\_ENSG00000185274|    

