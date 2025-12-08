**Repository Contents**

This is a package of reconstruction code for multiparametric mapping using non-water-suppressed MRSI, including:

(1) Main reconstruction functions: 

    -- "function_multiparametric_reconstruction.m": Multiparametric image reconstruction from sparse (k,t)-data

    -- [To-Do: Add deep learning function here]
    
    -- "function_GS_reconstruction.m": Generalized series-based spatial adaptation of the deep learning priors
    
(2) Support functions in folder "./support"

(3) Demo script: 

    -- "demo_multiparametric_water_recon.m"

<br>

**Data and Model Checkpoints**

Demo data can be downloaded from https://uofi.box.com/s/6g0bqoz6u2yydkny4evpvc92h30j503p

Deep learning model checkpoints can be downloaded from https://uofi.box.com/s/1lfh2ms7fuau95e3nypsosgvy5uzd46g

<br>

**System Requirements**

The reconstruction code requires a standard computer with enough RAM to process the multiparametric data.

The deep learning code requires [To-Do: Add system requirements for deep learning code]

This package is supported for the Linux operating system. The package has been tested on Linux Ubuntu 16.04 and Linux Ubuntu 18.04.

<br>

**Software Requirements**

The package has been tested on MATLAB R2014a and [To-Do: Add Python version here].

The list of dependent software required for deep learning model can be found in [To-Do: Add required envirenment file] and can be install by [To-Do: Add instruction on installing environment]

<br>


**Installation Instruction**

(1) Download the code package in this repo

(2) Download demo data and save it in the folder: './data/'

(3) Download deep learning model checkpoints and save it in [To-Do: Add instruction here]

(4) Run the demo script, type demo_multiparametric_water_recon in the MATLAB command line


