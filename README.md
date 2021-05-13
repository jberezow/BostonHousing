# BostonHousing
RJBNN project for the Boston Housing Data
-----------------------------------------

This repository concerns the implementation in Julia of Reversible Jump Markov Chain Monte Carlo (RJMCMC) on Bayesian Neural Network (BNN) models for the Boston Housing dataset. A regression task is performed on the popular dataset as a test of the predictive accuracy and trans-dimensional inferential capacity of a "Reversible Jump Bayesian Neural Network" (RJBNN) model.

Implementation
--------------

All code is writting in the Julia language and with the Gen probabilistic programming package.

Contents
========

Jupyter Notebooks
-----------------
Notebooks are used for exploratory programming (testing implementation of functions and Gen probabilistic models).

Julia Files
-----------
Julia files represent modules, functions or code that is used across multiple notebooks

Docker Folders
--------------
Experiments are run using containerization via Docker. The programs are defined along with all necessary model and algorithmic code in these folders. Containers are then built and uploaded to Docker hub for deployment to external HPC machines via Singularity.


More Information
================

A PDF copy of my thesis will be made available with this repository after completion. Further details about the components of the code will be included.
