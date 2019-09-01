===============
cynet
===============

.. figure:: https://img.shields.io/pypi/dm/cynet.svg
   :alt: cynet PyPI Downloads
.. figure:: https://img.shields.io/pypi/v/cynet.svg
   :alt: cynet version

.. image:: http://zed.uchicago.edu/logo/logozed1.png
   :height: 400px
   :scale: 50 %
   :alt: alternate text
   :align: center

.. class:: no-web no-pdf

:Info: See <https://arxiv.org/abs/1406.6651> for theoretical background
:Author: ZeD@UChicago <zed.uchicago.edu>
:Description: Implementation of the Deep Granger net inference algorithm, described in https://arxiv.org/abs/1406.6651, for learning spatio-temporal stochastic processes (*point processes*). **cynet** learns a network of generative local models, without assuming any specific model structure.

.. NOTE:: If issues arise with dependencies in python3, be sure that *tkinter* is installed

.. code-block::

    sudo apt-get install python3-tk

**Usage:**

.. code-block::

    from cynet import cynet
    from cynet.cynet import uNetworkModels as models
    from viscynet import viscynet as vcn

**cynet module includes:**
  * cynet
  * viscynet

cynet library classes:
~~~~~~~~~~~~~~~~~~~~~~
* spatioTemporal
* uNetworkModels
* simulateModels
* xgModels

Description of Pipeline:
  You may find two examples of this pipeline in your enviroment's bin folder
  after installing the cynet package.

Produces detailed timeseries predictions using Deep Granger Nets.

.. image:: detailed_Homicide01415.png
  :align: center

