Once you have installed cancerscope, you're set!   

Start by importing cancerscope into a python session like so (from command-line):
.. code:: bash  
    
    python

.. code:: python
    
    import cancerscope as cs

Tutorial with example file  
--------------------------

There is a tutorial with sample code, that walks through a small example of obtaining predictions directly from a file containing gene expression profiles. It's available on `Github <https://github.com/jasgrewal/cancerscope/tree/master/tutorial>` and is useful for understanding how the various data objects need to be structured in order to retrieve predictions from cancerscope.  

Predictions from cancerscope can be of two types. You can get the highest predicted category for each sample, with the accompanying confidence in prediction. You can also obtain a more granular decomposition of the prediction by retrieving a detailed view of the predictions. This view will enumerate the predicted cancer type(s) for each sample, alongwith the confidence for each category. It is useful if the machines in the ensemble are not in agreement about the highest predicted category.  

You can obtain predictions from gene expression profiles stored in a text file, in 3 easy steps. Here's the code, assuming each row is a samples and each column is a gene:

.. code-block:: python
    
    import cancerscope as cs
    scope_obj=cs.scope()
    preds_from_file = scope_obj.get_predictions_from_file("path/to/my/file_with_geneexpression.txt")

You can also load the data into the sample (row) * gene (column) form in python first, and then obtain predictions as follows:

.. code-block:: python
    
    import cancerscope as cs
    scope_obj=cs.scope()
    preds_df_from_xdat = scope_obj.predict(X = test_x, x_features = test_features, x_features_genecode = test_genecode, x_sample_names=test_samples)

Notice that with this approach, you'll need to provide other information like a list of gene names (*test_features*), a string indicating the type of gene name used (*test_genecode*), and optionally, a list of sample names (*test_samples*).


