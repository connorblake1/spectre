# The Spectre
# How to Use
Each of the following files should have its own configuration where you can run it: main.py, meta.py, superpack.py, phaseinverter.py, gifer.py, pdfbuilder.py, metasolve.py
translated.py holds only import statements and a wide range of aux. functions.

# meta.py
This is the most important file by far. It uses the generation algorithms to create 


# main.py
This file uses the spectre generation algorithms to directly create a n-order set of spectres. It has a primitive plotting function and can be used to look at large groups of vertices. It can also be used in conjunction with superpack.py to generate images superimposed onto each spectre such as to model a possible CO molecule packing.

# superpack.py
This uses some artificial "forcefields" to pack a number of circles into the outline of Spectre, possibly with some chiral edge modification. It then saves relevant files into RandomFiles that can be further manipulated into complete tilings with main.py.

# phaseinverter.py, gifer.py, pdfbuilder.py
These are used only for generating and manipulating images and have no bearing on the computational analysis.

# metasolve.py
This tested some old hypotheses that turned out to be really bad at capturing what was going on with the Spectre. Don't bother.

# Licenses
This code relies on scripts from Craig Kaplan's spectre generator at https://cs.uwaterloo.ca/~csk/spectre/app.html using a BSD licencse. I translated it from JS to Python. All other code is my own.

# Requirements
Packages: Numpy, Scipy, Matplotlib, Fitz, PIL, Pickle
