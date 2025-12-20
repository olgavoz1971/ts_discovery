# ts_discovery

This repository contains a set of notebooks in which I learn how to discover and properly download observational data using Virtual Observatory (VO) resources.

## scanned_plates Notebook

In this notebook we work with scanned photographic plates. Researchers sometimes face the task of tracing back the behaviour of a transient object, for example a cataclysmic variable star. For this purpose, archived photographic plates can be very useful.

The main chain of actions is:

1. Select a publishing registry TAP service
2. Search for data using the ObsCore table (it is crucial that the data are published via ObsCore)
3. Explore DataLink services
4. Download preview images and, if possible, execute server-side cutout services
5. *Make a great discovery, publish a paper, and receive a Nobel Prize*  
   *(beyond the scope of this notebook)*

