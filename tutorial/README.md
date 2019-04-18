# Tutorial

This tutorial will go through the use of cancerscope to predict the cancer type from a) an input file, or b) from pre-loaded RNA-Seq data.  
We will using some example RNA-Seq data from SRA. For this, you'll need the [pysradb](https://pypi.org/project/pysradb/) package.  
We will be using some example RNA-Seq data from TCGA. We will be downloading this data using the [gdc-rnaseq-tool]

[TCGA query](https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22%3E%3D%22%2C%22content%22%3A%7B%22field%22%3A%22cases.diagnoses.age_at_diagnosis%22%2C%22value%22%3A%5B6574%5D%7D%7D%2C%7B%22op%22%3A%22%3C%3D%22%2C%22content%22%3A%7B%22field%22%3A%22cases.diagnoses.age_at_diagnosis%22%2C%22value%22%3A%5B7304%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-HNSC%22%2C%22TCGA-LGG%22%2C%22TCGA-LIHC%22%2C%22TCGA-PCPG%22%2C%22TCGA-SKCM%22%2C%22TCGA-TGCT%22%2C%22TCGA-THCA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.access%22%2C%22value%22%3A%5B%22open%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.analysis.workflow_type%22%2C%22value%22%3A%5B%22HTSeq%20-%20FPKM%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22TXT%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Gene%20Expression%20Quantification%22%5D%7D%7D%5D%7D&searchTableTab=files)

## (Optional) Collating example data yourself  
The example data used was sourced as follows:  
`https://gdc.cancer.gov/access-data/gdc-data-transfer-tool`
`gdc-client download -m gdc_manifest_age22.txt`  

## Getting started
Please install cancerscope, and download all files in this directory.  

## Preparing Your Data  
When using cancerscope, you need the following 3 things: list of gene names, corresponding data matrix of gene RPKM or FPKM values, and a list of samples.  

These can be organized and passed in to cancerscope in two ways:  
1. They can be organized in a file that cancerscope reads in, like so:  

|ENSEMBL|SAMPLE1|SAMPLE2|Sample\_3\_with\_new\_naming\_style|SAMPLE4....|
|---|---|---|---|---|
|ENSG000012314|0.234|121|234|0.9823...|
|ENSG000058934|10.3452|234.2|2111\t0.245...|


Here, the GENE\_IDENTIFIER is "ENSEMBL"
...
`

If using a file orgnized like thus, the `get\_predictions\_from\_file(filename)` method for the scope object can be used to return a dictionary with the higest voted class from each machine in SCOPE.  

2. They can be organized separately as the data matrix, corresponding gene names list, and the GENE\_IDENTIFIER.  
- The input matrix has to be organized such that each row is a sample, and each column is a gene (note this is the transpose of the matrix in the file).  
- The `load_data(filename)` method of the scope object will take in the data file and return [X\_test, sample\_names\_orderedlist, features\_names\_orderedlist, gene\_code]  
The `predict` function in the scope object will take in X\_test, feature names (gene names, ordered), the gene identifier code, and optionally, x_sample_names.  

## Generating Predictions   
The SCOPE model can be set up by calling
`scope_test_obj = cancerscope.scope()`
This will set up references to the required models.  

Now you can pass in a test file and receive predictions, like so:  
`preds_df_from_file = scope_ensemble_obj.get_predictions_from_file(my_test_file)`  

Or alternatively, you can load in the data yourself and pass in the required information to `scope_ensemble_obj.predict()`.  
`preds_df_from_xdat = scope_ensemble_obj.predict(X = test_x, x_features = test_features, x_features_genecode = test_genecode, x_sample_names=test_samples)`   

Here, the passed objects look as follows:  
### test\_X  
`array([[  8.89943429e+00,   3.81203115e+01,   6.83880190e-02, ...,
          3.94528696e-01,   3.19967373e+00,   9.45898716e+00],
       [  4.73006431e+01,   5.64667745e+01,   1.02590420e+02, ...,
          6.84828984e+01,   8.59453759e+00,   1.10575758e+02]])
`  

The sahpe of test\_X , in this instance, is :  
`>>> test_X.shape()`

`(2, 19571)`  

This means the input had 2 samples, and 19571 genes.  

### test\_samples  
`['avg_tcga', 'fake_tcga']`

These are our two sample names, returned as a list of length 2.  

### test\_features  
`['HMGB1P1_ENSG00000124097', 'TIMM23_ENSG00000138297', ....]`  

This is a list of length 19571, containing feature names corresponding to the respective feature index in test\_X.  

### test\_genecode  
`SCOPE_ENSG`  
This is the GENE\_IDENTIFIER used in our example.  

### test\_samples  
``  

The default ouptut from both these functions is the top-voted class, accompanied by the average confidence score and contributing model names.  

### Tweaking the output format  
The output from the ensembl function cannot be modified much, unfortunately. You do have the option to save the predictions as:  

Alternatively, the output can be of the following formats. You need to set `ensemble\_score=False` before the system processes any of these flags. Otherwise they are set to `get_all_predictions=True, get_numeric=True`.  
 
i) The name of the top-voted class, for each model (default output from `cancerscope.scope.get_predictions_from_file()`, and from `cancerscope.scope.predict()`).  
ii) The probability score for the top-voted class, for each model (pass `get_numeric=True` to either of the two methods).  
iii) The probability scores for all the classes, unordered, for each model (pass `get_all_predictions=True, get_numeric=True` to either of the two methods).  
iv) The corresponding class labels for all the classes, corresponding to the probability scores in (iii), for each model (pass `get_all_predictions=True, get_numeric=False` to either of the two methods).  


### 


### FAQs  
What is the GENE\_IDENTIFIER?
The Gene Identifier has to be uniform across the list (don't provide a mix of ENSG, HUGO, and ENTREZ ids, for example). If passing in a file, the column name for the first column ('GENE\_IDENTIFIER') should indicate the type of genenames. Options are as follows:  
|SCOPE|ENTREZ|HUGO|GENENAME|ENSEMBL|HGNC|GSC1|GSC2|HUGO_ENSG|SCOPE_ENSG|
|---|---|---|---|---|---|---|---|---|---|
|A1BG|1|A1BG|A1BG|ENSG00000121410|5|A1BG\|1_calculated|merged_AIBG\|1\_calculated|A1BG_ENSG00000121410|A1BG_ENSG00000121410|  
|WBSCR17|64409|GALNT17|GALNT17|ENSG00000185274|16347|WBSCR17\|64409_calculated|merged_WBSCR17\|64409_calculated|GALNT17\_ENSG00000185274|WBSCR17\_ENSG00000185274|  

