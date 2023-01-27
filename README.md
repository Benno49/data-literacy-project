# Rotten Tomatoes Review Discrepancy
## Abstract
This project investigates the discrepancies between official critics' and the audience's ratings for movies on [Rotten Tomatoes](https://www.rottentomatoes.com/).  
We want to find out if there are any regularities and if these are enough to predict the scores (or their discrepancy/divergence).  
This project covers the data gathering and the evaluation with Null-Hypothesis Tests, Linear Regression, Decision Trees and MLPs.  
Results show that there is not enough information to infer any significant results.  
  
The full report is found [here](/src/presentation/Rotten_Tomatoes_Discrepancy.pdf).

## Project structure
### Data Collection
The Data is collected from Wikipedia and Rotten Tomatoes using [Scrapy](https://scrapy.org/).  
All unfiltered data is stored in [/src/movies/datasets](../../tree/main/src/movies/datasets).  

### Data Filtering and Processing
This part is covered in [/src/filter](../../tree/main/src/filter).  

#### Filter
The merging, filtering and converting process is performed in the notebook at [/src/filter/filter_dev.ipynb](/src/filter/filter_dev.ipynb).  

#### Statistics
The hypothesis tests are performed in [/src/filter/hypothesis_testing.py](/src/filter/hypothesis_testing.py).  
The applied learning is performed in [/src/filter/tomatoes.ipynb](/src/filter/tomatoes.ipynb).  

## Execution
It should suffice to pull this repository and install all requirements with 
```sh
python -m pip install -r requirements.txt
```
For the paths within the notebooks to work, one must set the root of the jupyter notebook by either navigating to the root of the repository and then executing the notebook or by using
```sh
jupyter notebook --notebook-dir=<path to repository>
```

## Information
This project was performed by
* Benjamin Raible  
* Daniel Kerezsy  
* Andreas Kotzur  

for the course **"Data Literacy (ML 4102)"** of the Eberhard Karls Universität Tübingen (2022/2023).
