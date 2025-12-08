**Repository Contents**

This is a package of reconstruction code for multiparametric mapping using non-water-suppressed MRSI, including:

(1) Main reconstruction functions: 

    -- "function_multiparametric_reconstruction.m": Multiparametric image reconstruction from sparse (k,t)-data

    -- "function_prepare_superRes_input/output.m"+"run_[T1map/PDmap/T2map]_I2SB_SR_demo.py": Learning-based super-resolution reconstruction
    
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

Hardware Requirement: A standard workstation with sufficient CPU/GPU RAM

Supported Operating Systems: Linux (tested on Ubuntu 16.04 and Ubuntu 18.04)

<br>

**Software Requirements**

MATLAB R2014a (fully tested)

Python 3.10.4

Deep learning dependencies: env_diffusion.yml

<br>


**Installation Instruction**

(1) Download the code package in this repo

(2) Install the deep learning dependencies: [To-Do: Add instruction here]

(3) Download demo data and save it in the folder: './data/'

(4) Download deep learning model checkpoints and save it in [To-Do: Add instruction here]

(5) Run the demo script, type demo_multiparametric_water_recon in the MATLAB command line


