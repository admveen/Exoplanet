Exoplanet Discovery using the Light Curve Transit Method
==============================



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
    ├── reports            <- Presentation and final report
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    |   ├── presentation   <- Presentation + beamer LateX generating files
    |   |    └── presentation.pdf   <- Presentation created by beamer LateX
    |   |
    |   ├──final_report.pdf <-- Final report. 
    |   └──model_metrics.md <-- Markdown containing classification reports for best models. 
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