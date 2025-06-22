import streamlit as st
import geopandas as gpd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

@st.cache_data
def load_co2_data():
    path = r"G:\CNN_LSTM\predicted_CO2_2034.nc"
    ds = xr.open_dataset(path)
    return ds

@st.cache_data
def load_country_shapes():
    shp_path = r"G:\CNN_LSTM\ne_110m_admin_0_countries (4)\ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shp_path)
    return world

def plot_co2_exact(ds, country_geom, selected_country, time_idx):
    co2_data = ds['predicted_CO2'].isel(time=time_idx)

    country_gdf = gpd.GeoDataFrame(geometry=country_geom)
    country_gdf = country_gdf.set_crs("EPSG:4326")

    co2_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    co2_data.rio.write_crs("EPSG:4326", inplace=True)

    masked = co2_data.rio.clip(country_gdf.geometry, country_gdf.crs, drop=True)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    masked.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        cbar_kwargs={"label": "CO₂ (ppm)"}
    )

    country_gdf.boundary.plot(ax=ax, edgecolor="blue", linewidth=1.5)

    ax.set_title(f"{selected_country} — Predicted CO₂ (Year Index: {time_idx})")
    ax.set_extent([
        float(masked.lon.min()), float(masked.lon.max()),
        float(masked.lat.min()), float(masked.lat.max())
    ])
    ax.coastlines(resolution='50m')

    st.pyplot(fig)

st.set_page_config(layout="wide")
st.title("Future CO₂ Prediction App")

ds = load_co2_data()
world = load_country_shapes()

countries = world['NAME'].unique()
selected_country = st.selectbox("Select a country", sorted(countries))

country_geom = world[world['NAME'] == selected_country].geometry

time_len = ds.dims['time']
time_idx = st.slider("Select year index (0 = 2026)", 0, time_len - 1, 0)

plot_co2_exact(ds, country_geom, selected_country, time_idx)
