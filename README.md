[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YlfKWlZ5)

# Equity and Efficiency in Service Delivery: Evidence from 311 Requests in Chicago

This project analyzes the provision of services in Chicago using individual-level 311 Service Requests data.

The policy topic centers on equity and efficiency in local service delivery, examining whether neighborhoods with different characteristics exhibit systematic variation in the composition of service requests and in request outcomes. By linking service requests to neighborhood characteristics, the project explores whether disparities in municipal service provision and responsiveness are associated with socioeconomic and demographic differences across Chicago neighborhoods.

## Setup

```bash
conda env create -f environment.yml
```

## Project Structure

```
data/
  raw-data/                       # Raw data files
    community_area.csv            # Chicago community area boundaries and metadata used for spatial joins and geographic aggregation
    311_request.csv               # Raw 311 service request records used for complaint analysis and filtering (available here: https://drive.google.com/drive/folders/1xwaIqXUsbTgMgTnBrFVNo45hhgikIauK)
  derived-data/                   # Filtered data and output plots
    community_area_filtered.gpkg  # Spatial dataset filtered to study period and prepared for mapping (GeoPackage format)
    311_filtered.csv              # Cleaned and filtered 311 data used for visualization and analysis
code/
  preprocessing.py                # Filters community area and 31 requests data
  static_plots.py                 # Plots 311 requests + community area perimeters
```

## Usage

1. Run preprocessing to filter data:
   ```bash
   python code/preprocessing.py
   ```

2. Generate the acs perimeter plot:
   ```bash
   python code/plot_acs.py  
   ```

## Requirements
gdown

