Exoplanet Discovery using the Light Curve Transit Method
==============================

Introduction
------------
In 1992 astronomers discovered a periodic set of dips modulating the emission from a pulsating neutron star. The source of these dips was identified as two planetary bodies orbiting the neutron star. The possibility of discovering and examining planets that do not orbit our sun was a compelling one. Some relevant reasons include the desire to find Earth-similar planets that could host extra-terrestrial life, or to understand the distribution of the types of planets and planetary systems that exist and where our own solar system fits in this schema.

This project focuses on the identification of exoplanets from stellar light curve data using the transit detection method. Stellar light curves are time series of the measured intensity of a target star. Exoplanets are too small to see directly by telescope but an exoplanet transiting into the line of sight between the target star and space telescope will cause a small dip in the measured stellar intensity. 
![alt text](reports/figures/transit_illustration.jpg?v=4&s=200)
If the time series is long enough, the transits of the orbiting planet create light intensity dips that are observed to be periodic. An example light curve of a real exoplanet is shown below. 
![alt text](reports/figures/exo_multiple_transit.jpg?v=4&s=200)

The vast majority of observed light curves do not have significant transiting events. A first round of statistical tests can be used to ascertain whether there are transiting events. But there are light curves with statistically significant transiting events that do not correspond to actual exoplanets. Some of these false positives are eclipsing binary star systems. False positives from eclipsing binary systems have specific light curve characteristics -- particularly the presence of secondary smaller-amplitude dips as the brighter star eclipses a dimmer binary partner. We denote these as secondary eclipse false positives. An example of such a light curve can be seen below:
![alt text](reports/figures/algol-curve.png?v=4&s=200)
Other types of false positives are variable stars with periodic pulsation in their luminosity, artifacts due to polluting light from nearby stars or just plain junk. These are denoted as non-transiting phenomena false positives. <br>

**The main task of this study is to train a classifier that can separately identify real exoplanets, secondary eclipse false positives, and non-transiting false positives from an analysis of the light curves.** <br>

Data Sources 
------------
We used light curves from the Kepler mission. The Kepler space telescope observed approximately 500,000 stars in a portion of the Milky Way galaxy from 2009-2018. A small portion of the stars measured had light curves with statistically significant events. These objects were flagged and entered into a Kepler Object of Interest (KOI) catalog for further analysis. We focus on these KOIs for our classification task

Metadata and extracted parameters for KOIs were taken from the Kepler cumulative table via API requests to the NASA Exoplanet archive: <br>
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative' <br>
Light curve data was taken from the Minkulski Archive for Space Telescopes (MAST) via their exoplanet mission API (the exoMAST API). Documentation on interacting with the API can be found here. <br>
https://exo.mast.stsci.edu/docs/ <br>

Summary 
------------
A high level description of the steps we followed in data wrangling, EDA, preprocessing, and modeling/evaluation can be found in the final report [here](https://github.com/admveen/Exoplanet/blob/master/reports/final_report.pdf).

A slide deck containing much of this information is also [here](https://github.com/admveen/Exoplanet/blob/master/reports/presentation/presentation.pdf).

Further detail of the analysis can be found in the notebooks and in certain relevant source files:

<h2>I. Data wrangling  <h2 />

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data downloaded / wrangled from APIs
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │   
    │
    ├── docs               <- Contains the initial project proposal.
    │
    ├── models             <- contains three best models (pickled) found by hyperparameter tuning.
    │   └── preprocessing  <- contains pickled preprocessing/data transformation pipeline.
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is Exo_DSMstep_partnumber
    |                         DSMstep <--> DataWrangling, EDA, Preprocessing, or Modeling
    |                         partnumber <--> relevant when a given DSM step is split into parts.
    │                         
    ├── references         <- Journal references
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    |   ├── presentation   <- Presentation + beamer LateX generating files
    |   |    └── presentation.pdf   <- Presentation created by beamer LateX
    |   |
    |   ├──final_report.pdf <-- Final report. 
    |   └──model_metrics.md <-- Markdown containing classification reports for best models. 
    |
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── KOIclass.py    <- Contains custom-built class that grapples with the ExoMAST API,
    │   │                      NASA Exoplanet archive API, does most of the time-series processing,
    │   │                      has various light curve plotting functions, performs feature construction,
    │   │                      and controls light curve downloading for a single KOI.
    |   ├── exo_preprocess.py   <- data preprocessing used by automated report generator and image generator
    |   |                          python source files
    │   │
    │   ├── data           <- Script to download 26 GB of light curves for relevant KOIs
    │   │   └── make_dataset.py  <- Downloads light curves into data/external/DVSeries
    │   │
    │   │
    │   ├── models         <- Scripts to output model report markdown
    │   |    └── construct_modelreports.py
    │   │   
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── construct_eda_images_pt.py   <- Light curve visualizations for final report
    │       ├── construct_eda_images_pt2.py  <- Statistical EDA images for final report
    │       └── construct_eda_images_pt2_presentaton.py  <- Statistical EDA images for presentation
    │
    └── 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
