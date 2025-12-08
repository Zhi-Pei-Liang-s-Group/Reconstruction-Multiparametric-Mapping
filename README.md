This is a package of reconstruction code for multiparametric mapping using non-water-suppressed MRSI, including:

(1) Main reconstruction functions: 

    -- "function_multiparametric_reconstruction.m": Multiparametric image reconstruction from sparse (k,t)-data
    
    -- "function_GS_reconstruction.m": Generalized series-based spatial adaptation of the deep learning priors
    
(2) Support functions in folder "support"

(3) Demo script: demo_multiparametric_water_recon.m

<br>

Demo data can be downloaded from https://uofi.box.com/s/6g0bqoz6u2yydkny4evpvc92h30j503p

Deep learning model weight checkpoints can be downloaded from https://uofi.box.com/s/1lfh2ms7fuau95e3nypsosgvy5uzd46g

<br>

System requirements:
The reconstruction code requires a standard computer with enough RAM to process the multiparametric data.
This package is supported for the Linux operating system. The package has been tested on Linux Ubuntu 16.04 and Linux Ubuntu 18.04.

<br>

Software requirements:
The package has been tested on MATLAB R2014a.

<br>

Setup:
All dependencies have been included in the package.

<br>

Demo:
To run the demo script, type demo_multiparametric_water_recon in the MATLAB command line.
