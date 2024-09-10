import contextily as cx
import geopandas
import h3
import h3pandas
import rioxarray
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import shapely
import statsmodels.api as sm

from pathlib import Path
from pyproj import Transformer
from typing import Optional, Union


def raster_to_h3(layer_file: Union[str, Path], column_name: str, h3_resolution: int) -> geopandas.GeoDataFrame:
    # load raster to xyz dataframe
    df = (
        rioxarray.open_rasterio(layer_file)
        .rio.reproject("EPSG:4326")
        .sel(band=1)
        .to_pandas()
        .stack()
        .reset_index()
        .rename(columns={"x": "lng", "y": "lat", 0: column_name})
    )
    # ignore background values
    df = df[df[column_name] != 0]

    # find hexs containing the points
    # raster values are summed for each h3 cell
    hex_col = "h3_" + str(h3_resolution)
    df[hex_col] = df.apply(lambda x: h3.geo_to_h3(x.lat, x.lng, h3_resolution), 1)
    df_h3 = df.groupby(hex_col)[column_name].sum().to_frame(column_name).reset_index()

    # build h3 geometry and subset columns
    geom = [
        shapely.geometry.Polygon(h3.h3_to_geo_boundary(row[hex_col], geo_json=True)) for index, row in df_h3.iterrows()
    ]
    gdf_h3 = geopandas.GeoDataFrame(df_h3, geometry=geom, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_h3 = gdf_h3[[hex_col, column_name, "geometry"]]
    return gdf_h3


def vector_to_h3(
    layer_file: Union[str, Path],
    column_name: str,
    h3_resolution: int,
    bbox: Optional[
        Union[tuple[float, float, float, float], geopandas.GeoDataFrame, geopandas.GeoSeries, shapely.Geometry]
    ] = None,
) -> geopandas.GeoDataFrame:
    # load vector to geodataframe
    gdf = geopandas.GeoDataFrame.from_file(layer_file, bbox=bbox).to_crs(epsg=4326)
    geom_type = gdf.geom_type[0]
    hex_col = "h3_" + str(h3_resolution)
    if geom_type == "Polygon":
        # find hexs containing the polygons and build h3 geometry
        # polygon values are assigned directly to the containing h3 cells
        gdf_h3 = gdf.h3.polyfill_resample(resolution=h3_resolution, return_geometry=True).to_crs(epsg=3857)
        index_name = gdf_h3.index.name
        gdf_h3 = gdf_h3.reset_index().rename(columns={"value": column_name, index_name: hex_col})
    elif geom_type == "Point":
        # find hexs containing the points and build h3 geometry
        # point values are summed up to the containing h3 cells
        gdf_h3 = gdf.h3.geo_to_h3_aggregate(resolution=h3_resolution, operation="sum", return_geometry=True).to_crs(
            epsg=3857
        )
        index_name = gdf_h3.index.name
        gdf_h3 = gdf_h3.reset_index().rename(columns={"value": column_name, index_name: hex_col})
    else:
        raise NotImplementedError(f"geometry type of layer {layer_file.stem} not supported [Point, Polygon]")
    # subset columns
    gdf_h3 = gdf_h3[[hex_col, column_name, "geometry"]]
    return gdf_h3


def loglinear_pooling(
    weights: tuple[float, ...], p_layers: tuple[geopandas.GeoDataFrame, ...]
) -> geopandas.GeoDataFrame:
    with np.errstate(divide="ignore"):
        # loglinear probability pooling
        p_pooled = np.zeros_like(p_layers[0])
        for idx, p_layer in enumerate(p_layers):
            p_layer /= np.sum(p_layer)
            p_pooled += weights[idx] * np.log(p_layer)
        p_pooled = np.exp(p_pooled)
    # normalize results
    lo, hi = np.quantile(p_pooled, 0.0), np.quantile(p_pooled, 1.0)
    p_pooled = np.clip((p_pooled - lo) / (hi - lo), a_min=0.0, a_max=1.0)
    return p_pooled


def kl_divergence_sum(parameter: dict, p_layers: tuple[geopandas.GeoDataFrame, ...]) -> pd.DataFrame:
    pooling_weights = []
    for k in parameter.keys():
        if "pooling_weights" in k:
            pooling_weights.append(parameter[k])
    if 0.0 in pooling_weights:
        pass
    else:
        # compute sum of Kullback Leibler Divergence between input layers and prediction
        p_pooled = loglinear_pooling(pooling_weights, p_layers)
        p_pooled += 1e-8
        kl_sum = 0
        for p_layer in p_layers:
            p_layer += 1e-8
            kl_sum += np.sum(sp.special.kl_div(p_layer, p_pooled))
        kl_sum_df = pd.DataFrame.from_dict(
            {
                "pooling_weights": [pooling_weights],
                "kl_sum": [kl_sum],
            }
        )
        return kl_sum_df


def regression_metrics(
    x: geopandas.GeoSeries, y: geopandas.GeoSeries
) -> tuple[float, float, float, float, float, float]:
    x = sm.add_constant(x)
    model = sm.OLS(endog=y, exog=x)
    res = model.fit()
    r_square_adj = round(float(res.rsquared_adj), 3)
    coeff = round(float(res.params.iloc[1]), 3)
    p = round(float(res.pvalues.iloc[1]), 3)
    aic = round(float(res.aic), 3)
    bic = round(float(res.bic), 3)
    resid = res.resid.rename("residuals").to_frame()
    return r_square_adj, coeff, p, aic, bic, resid


def plot_gdf_basemap(
    gdf: geopandas.GeoDataFrame,
    column: str,
    cmap: Optional[str] = "Reds",
    bounds: Optional[tuple[float, float, float, float]] = None,
    add_labels: Optional[bool] = True,
    center_cmap: Optional[bool] = False,
    crs: Optional[int] = 3857,
) -> plt:
    fig, ax = plt.subplots()
    if bounds is not None:
        transformer = Transformer.from_crs(gdf.crs, crs, always_xy=True)
        x1, y1 = transformer.transform(bounds[0], bounds[1])
        x2, y2 = transformer.transform(bounds[2], bounds[3])
        bounds = [x1, y1, x2, y2]
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

    gdf = gdf.to_crs(crs)

    if center_cmap:
        # center the colormap to zero
        # only useful if colormap is diverging (e.g. difference map)
        gdf.plot(alpha=1.0, column=column, cmap=cmap, legend=True, ax=ax, norm=mpl.colors.CenteredNorm())
    else:
        # use colormap as is (the default use case)
        gdf.plot(alpha=1.0, column=column, cmap=cmap, legend=True, ax=ax)
    providers = cx.providers.flatten()
    cx.add_basemap(
        ax,
        crs=gdf.crs.to_string(),
        interpolation="sinc",
        zoom="auto",
        attribution="",
        source=providers["CartoDB.PositronNoLabels"],
    )
    if add_labels:
        cx.add_basemap(
            ax,
            crs=gdf.crs.to_string(),
            interpolation="sinc",
            zoom="auto",
            attribution="",
            source=providers["CartoDB.PositronOnlyLabels"],
        )
    ax.set_title(f"{column}")
    plt.tight_layout()
