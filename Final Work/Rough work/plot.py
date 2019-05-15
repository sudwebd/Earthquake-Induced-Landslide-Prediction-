import geopandas as gpd
import geoplot as gplt
import pandas as pd; pd.set_option('max_columns', 6)

# Load and display some example data.
# All of the examples in this notebook use the `quilt` package to do this.
from quilt.data.ResidentMario import geoplot_data
continental_cities = gpd.read_file(geoplot_data.usa_cities()).query('POP_2010 > 100000')
contiguous_usa = gpd.read_file(geoplot_data.contiguous_usa())
continental_cities.head()

import matplotlib.pyplot as plt

collisions = gpd.read_file(geoplot_data.nyc_collision_factors())
boroughs = gpd.read_file(geoplot_data.nyc_boroughs())
census_tracts = gpd.read_file(geoplot_data.ny_census_partial())
percent_white = census_tracts['WHITE'] / census_tracts['POP2000']
obesity = geoplot_data.obesity_by_state()
contiguous_usa = gpd.read_file(geoplot_data.contiguous_usa())
contiguous_usa['Obesity Rate'] = contiguous_usa['State'].map(
    lambda state: obesity.query("State == @state").iloc[0]['Percent']
)
la_flights = gpd.read_file(geoplot_data.la_flights())
la_flights = la_flights.assign(
    start=la_flights.geometry.map(lambda mp: mp[0]),
    end=la_flights.geometry.map(lambda mp: mp[1]))
collisions = gpd.read_file(geoplot_data.nyc_collision_factors())
boroughs = gpd.read_file(geoplot_data.nyc_boroughs())

ax = gplt.kdeplot(collisions, 
                  projection=gcrs.AlbersEqualArea(), 
                  shade=True,  # Shade the areas or draw relief lines?
                  shade_lowest=False,  # Don't shade near-zeros.
                  clip=boroughs.geometry,  # Constrain the heatmap to this area.
                  figsize=(12,12))
gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), ax=ax)
