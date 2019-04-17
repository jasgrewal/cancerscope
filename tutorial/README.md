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
`
GENE_IDENTIFIER*\tSAMPLE1\tSAMPLE2\tSAMPLE_3_with_new_style\tSAMPLE4....
ENSG000012314\t0.234\t121\t234\t0.9823...
ENSG000058934\t10.3452\t234.2\t2111\t0.245...
...
`

If using a file orgnized like thus, the `get\_predictions\_from\_file(filename)` method for the scope object can be used to return a dictionary with the higest voted class from each machine in SCOPE.  

2. They can be organized separately as the data matrix, corresponding gene names list, and the GENE\_IDENTIFIER.  
- The input matrix has to be organized such that each row is a sample, and each column is a gene (note this is the transpose of the matrix in the file).  
- The `load\_data(filename)` method of the scope object will take in the data file and return [X\_test, sample\_names\_orderedlist, features\_names\_orderedlist, gene\_code]  
The `predict` function in the scope object will take in X\_test, feature names (gene names, ordered), and the gene code.  

## Generating Predictions   
The SCOPE model can be set up by calling
`scope_test_obj = cancerscope.scope()`
This will set up references to the required models.  

The default ouptut from both these functions is the top-voted class, accompanied by the average confidence score and contributing model names.  

### Tweaking the output format  
Alternatively, the output can be of the following formats. You need to set `ensemble\_score=False` before the system processes any of these flags. Otherwise they are set to `get_all_predictions=True, get_numeric=True`.  
 
i) The name of the top-voted class, for each model (default output from `cancerscope.scope.get_predictions_from_file()`, and from `cancerscope.scope.predict()`).  
ii) The probability score for the top-voted class, for each model (pass `get_numeric=True` to either of the two methods).  
iii) The probability scores for all the classes, unordered, for each model (pass `get_all_predictions=True, get_numeric=True` to either of the two methods).  
iv) The corresponding class labels for all the classes, corresponding to the probability scores in (iii), for each model (pass `get_all_predictions=True, get_numeric=False` to either of the two methods).  


### 


### FAQs  
What is the GENE\_IDENTIFIER?
The Gene Identifier has to be uniform across the list (don't provide a mix of ENSG, HUGO, and ENTREZ ids, for example). If passing in a file, the column name for the first column ('GENE\_IDENTIFIER') should indicate the type of genenames. The value of this variable includes HUGO, ENSEMBL, ENTREZ.        


