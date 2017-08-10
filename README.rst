contrastive
===================
A python3 library for performing unsupervised machine learning on datasets with learning (e.g. PCA) in contrastive settings, where one is interested in patterns (e.g. clusters or clines) that exist one dataset, but not the other.

Applications include dicovering subgroups in biological and medical data.

Installation
--------------------

.. code-block:: 

	$ pip install contrastive

Examples
-------------------------------

The basic functions enabled by this library are shown below. Generally speaking, we have two datasets, one is a dataset that we can label as  :code:`foreground_data`, which is the dataset in which we are discovering patterns and directions, and another dataset called :code:`background_data`, which is the dataset that does not have the patterns or directions we are interested in discovering. In some cases, both datasets may contain the signal of interest, but the foreground dataset may have the pattern enriched relative to the background. In these analyses, there is a contrast parameter, known as alpha, which can be thought of as a hyperparameter.

.. code-block:: python

	from contrastive import CPCA

	mdl = CPCA()
	projected_data, alphas = mdl.fit_transform(foreground_data, background_data)
	
	#returns a set of 2-dimensional projections of the foreground data stored in the list 'projected_data', for several different values of 'alpha' that are automatically chosen (by default, 4 values of alpha are chosen)

Built-in plotting: to quickly see the results of PCA, simply enable the :code:`plot` parameter to true:

.. code-block:: python

	from contrastive import CPCA

	mdl = CPCA()
	projected_data, alphas = mdl.fit_transform(foreground_data, background_data, plot=True)
	

Interactive GUI: if you are running these analyses inside a jupyter notebook, you can easily launch an interactive GUI as shown here:

.. code-block:: python

	from contrastive import CPCA

	mdl = CPCA()
	projected_data, alphas = mdl.fit_transform(foreground_data, background_data, gui=True)
	
