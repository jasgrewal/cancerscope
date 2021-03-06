Installation
-------------

Cancerscope is available through pip. It supports all Python versions upwards of Python 2.7. It has been developed and tested with Python 2.7.14, 3.4, 3.5, 3.6, and 3.7.  

.. code:: bash
    pip install cancerscope  

Installing previous releases
-----------------------------

To upgrade to a newer release, simply run the following bash command:
.. code:: bash
    pip install --upgrade cancerscope

To install a previous release, for example, v.0.42 (last supported release that used theano and lasagne):
.. code:: bash
    pip install cancerscope=0.42

Installing bleeding edge release  
---------------------------------
To install the latest release instead of the most recent stable release, you can install from source. Use the following bash commands:

>>> git clone https://github.com/jasgrewal/cancerscope.git
>>> cd cancerscope
>>> python setup.py install

Major version changes
---------------------
Model setup using theano and lasagne
.. versionchanged:: 0.42

Model setup using keras  
.. versionadded:: 1.0  

Plotting option available for python 2.7
.. versionadded:: 0.0  

Plotting option removed
.. versionchanged:: 0.31  


