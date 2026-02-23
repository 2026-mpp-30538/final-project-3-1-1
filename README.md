[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YlfKWlZ5)

# Equity and Efficiency in Service Delivery: Evidence from 311 Requests in Chicago

This project analyzes the provision of services in Chicago using individual-level 311 Service Requests data. 

The policy topic centers on equity and efficiency in local service delivery, examining whether neighborhoods with different characteristics exhibit systematic variation in the composition of service requests and in request outcomes. By linking service requests to neighborhood characteristics, the project explores whether disparities in municipal service provision and responsiveness are associated with socioeconomic and demographic differences across Chicago neighborhoods..

## Setup

```bash
conda env create -f environment.yml
conda activate fire_analysis
```

## Project Structure

```
data/
  raw-data/           # Raw data files
    fire.csv          # Historical fire perimeter data
    canadian_cpi.csv  # Canadian Consumer Price Index data
  derived-data/       # Filtered data and output plots
    fire_filtered.gpkg  # Fire data filtered to post-2015
    cpi_filtered.csv    # CPI data filtered to 2020 onwards
code/
  preprocessing.py    # Filters fire and CPI data
  plot_fires.py       # Plots fire perimeters
```

## Usage

1. Run preprocessing to filter data:
   ```bash
   python code/preprocessing.py
   ```

2. Generate the fire perimeter plot:
   ```bash
   python code/plot_fires.py
   ```
