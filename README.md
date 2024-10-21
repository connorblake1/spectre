# The Spectre
See TheSpectre9.12.23.pdf for a writeup.
# Some Results
The code can be used to generate arbitrarily sized "patches" of metatiles and display them in various forms. Shown here, in order, are "raw vertex", "colored metatile", and "pairs forms".

![alt text](https://github.com/connorblake1/spectre/blob/master/rawvertex.png?raw=true)
![alt text](https://github.com/connorblake1/spectre/blob/master/metatiles.png?raw=true)
![alt text](https://github.com/connorblake1/spectre/blob/master/pairs.png?raw=true)

The code is also able to generate each eigenstate for each patch and display them via both tight-binding and Landau-gauge Hamiltonians.

![alt text](https://github.com/connorblake1/spectre/blob/master/tb.png?raw=true)
![alt text](https://github.com/connorblake1/spectre/blob/master/tb2.png?raw=true)
![alt text](https://github.com/connorblake1/spectre/blob/master/landau.png?raw=true)
An explanation of the notation and graphs can be found in the provisional writeup in this repo.

# How to Use
Each of the following files should have its own configuration where you can run it: main.py, meta.py, superpack.py, phaseinverter.py, gifer.py, pdfbuilder.py, metasolve.py, oldmain.py. translated.py holds only import statements and a wide range of aux. functions.

## meta.py
This is the most important file by far. It uses the generation algorithms to create a tiling of hexagons. The user then selects a patch of these tiles to isolate with iList. After changing the required settings in the main hyperparameter section, the code generates a set of metatiles defined by iList/ind_list, zips all the adjacent vertices together, and runs the selected calculations. This program has high-quality graph generation features in drawStates(), drawSpectra(), and drawIndivState(). In the hyperparameter section, there are a large set of booleans and short lists that can be changed by the user to run certain subroutines. Each one has an explanation for how to use it. They functions defined here broadly fall into 2 categories: Terminal and Workflow. Both Terminal and Workflow functions are further classified into Passthrough, Print, and Save. They are labelled within the code accordingly with instructions for use. All hyperparameters a user should change fall within the Hyperparameters and End Hyperparameters comments unless you want to go insane.

### Terminal
Terminal computations exit the program upon completion and are typically small side-computations that ride on the main framework. These include computing a simulated diffraction grating (and saving it), and simulating homogeneous hexagonal (graphene) or triangular lattices. All either Print outputs to the console or Save files to the directory in a labelled folder.

### Workflow
Workflow functions run calculations that often build off each other and may require multiple active settings (ie you have to make multiple items True to work). These run the core computations that require a set of metatiles and their vertices. Some hyperparameters slightly modify other functions (like saveHexType modifying showHex) while others may generate hundreds of files over multiple hours if requested (drawStates, depending on settings).

## main.py
This file uses the spectre generation algorithms to directly create a n-order set of spectres. It has a primitive plotting function and can be used to look at large groups of vertices. It can also be used in conjunction with superpack.py to generate images superimposed onto each spectre such as to model a possible CO molecule packing.

## superpack.py
This uses some artificial "forcefields" to pack a number of circles into the outline of Spectre, possibly with some chiral edge modification. It then saves relevant files into RandomFiles that can be further manipulated into complete tilings with main.py.

## phaseinverter.py, gifer.py, pdfbuilder.py
These are used only for generating and manipulating images and have no bearing on the computational analysis.

## metasolve.py, oldmain.py
This tested some old hypotheses that turned out to be really bad at capturing what was going on with the Spectre. Don't bother.

# Licenses
This code relies on scripts from Craig Kaplan's spectre generator at https://cs.uwaterloo.ca/~csk/spectre/app.html using a BSD licencse. I translated it from JS to Python. All other code is my own.

# Requirements
Packages: Numpy, Scipy, Matplotlib, Fitz, PIL, Pickle
