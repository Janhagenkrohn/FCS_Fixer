# FCS_Fixer

FCS_Fixer is a Python module meant to make data analysis for Fluorescence Correlation Spectroscopy (FCS) data acquired in time-tagged, time-resolved mode, easier. 

In FCS experiments, one often deals not only with the desired signal, but also various artifacts, some sample-related, some instrument-related. Such artifacts include (but are not limited to):
1. High-intensity "Bursts" from bright particles moving through the detection volume
2. Signal drift from photobleaching (or from z-drift in the case of membrane FCS experiments)
3. Background from ambient light, detector dark current, sample autofluorescence, etc.
4. Detector afterpulsing

FCS_Fixer attempts to fix the mentioned artifacts in a largely automated manner. "Automation" here refers to the fact that the user only needs to...
1. Specify which data to use
2. Define which filters to use (where the user can choose freely the order in which to chain multiple of them)
3. Supply some metaparameters to the filters that are used for statistical criteria by which to chose the actual filter parameters (default values that in our hands work well for many settings are set in the background)

However, the module also leaves plenty of backdoors to access the parameters of some filter functions on a more direct level, or even combine the built-in filters and logic with entirely custom weighting, time gating, or photon selection functions.

To inspect what exact filters are implemented, check out the Jupyter Notebook 01_overview,ipynb. The following notebooks and scripts then show how to use the module with increasingly simplified user input, standardizing the data pipeline more and more for rapid use in batch processing.

Note that FCS_Fixer is compatible with Fluorescence Cross-Correlation spectroscopy, Pulsed Interleaved Excitation, and filtered FCS/Fluorescence Lifetime Correlation Spectroscopy as well. While in principle data from Scanning FCS etc. can be processed as well, there are currently no routines for creating the spatial correlation maps frequently used in these techniques. 


## Strengths
- Implementation of various published strategies for artifact removal in FCS, with slight modifications. Not restricted to specific FCS variants.
- High degree of automation of parameter choice, allowing a "click and forget" solution for many datasets.
- User can freely customize the the order in which all filters are to be applied, and has access to many options to tune parameters or otherwise customize for more complex data.
- Simple use requiring small number of class/function calls in Python scripts or Jupyter Notebooks.
- Fully Python-based implementation makes extension/customization easy.
- Automated export of:
	1. .csv files containing spreadsheets with many intermediates and results. Correlation functions in particular are written in the 4-column Kristine format used by the software of Claus Seidel's lab. The user can choose whether to export the .csv files with or without header lines.
	2. .png figure files. Every .csv file is accompanied by a .png file visualizing the content of the .csv file.
	3. log file with detailled recording of most data processing steps, allowing the user to trace relatively easily what statistical criteria made the software make what decisions, as well as listing some summary statistics on how the data responded to the processing steps.

## Weaknesses
- No Graphical User Interface.
- Tested only in Windows environment, and only on PicoQuant .ptu data.
- Tested for Conda environments built around Python 3.10 and tttrlib 0.26
- No support for FCS raw data formats that are not compatible with tttrlib.
- Fully Python-based implementation means that **performance is relatively low**. In fact, we are aware that various methods could be more efficient even with pure Python, but we opted against that, sometimes for reasons of flexibility, sometimes simply because weI prioritized moving forward with development over writing high-performance code. We may improve on some of these later on.


## Installation
Currently, actual installation of FCS_Fixer into a Python environment is not set up. Instead, we explicitly import it from the directory of the repo. We may look into that at a later point....

For now, let's look at how to set up the correct Python environment. Use the Anaconda distribution of Python. The pipeline described here is currently tested in Windows environments, but there is a good chance it should work on Linux as well.

The compatibility is currently ensured for a typical [theatRICS](https://github.com/yusuf-qutbuddin/theatRICS) conda environment.

### Create environment
In the command shell, run:

`conda create --name tttr python=3.10` 

`tttr` is just the environment name we use, you may replace that with whatever you like.

### Install required packages
Activate the environment and install the required packages:

`conda activate tttr`

`pip install theatrics`

Installing theatRICS will also install all the dependencies for FCS_Fixer, nothing else is needed.

Additionally, you'll want to install Spyder or another IDE:

`conda install spyder`

### Getting FCS_Fixer itself
Simply clone the GitHub repo into some local directory on your machine, and you're good to go. For testing, `cd` into your local copy of the repo. Inside the repo, try to run one of the Jupyter Notebooks. The notebook 01_overview.ipynb also explains how to import the module for executing code.

Typing `jupyter notebook` into the shell should open Jupyter Notebook inside a browser. If not, look into information on Jupyter Notebook errors on the web, we cannot predict here what errors may cause this.

One known issue Jupyter Notebook we'd like to mention here as we encountered it ourselves is that sometimes, depending on the exact history of how you handle your conda environments, Jupyter Notebook ends up trying to run a different Python version than what the environment. In that case, fix the issue by running these two commands while in the `tttr` environment:

`pip install ipykernel`

`python -m ipykernel install --user`


### Development
The module is built around tttrlib (https://github.com/Fluorescence-Tools/tttrlib), and some functions also use code snippets taken from tttrlib application examples.

FCS_Fixer core developer and contact for inquiries: Jan-Hagen Krohn (jan-hagen.krohn@uk-essen.de)

Assistance in pipeline development and debugging: Béla Frohn, Lise Isnel, Yusuf Qutbuddin

