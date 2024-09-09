import geopandas
import h3
import h3pandas
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import scipy as sp

from joblib import Parallel, delayed
from pathlib import Path
from shapely.geometry import Polygon
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from .utils import plot_gdf_basemap


def loglinear_pooling(weights, p_layers):
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


def kl_divergence_sum(parameter, p_layers):
    pooling_weights = []
    for k in parameter.keys():
        if "pooling_weights" in k:
            pooling_weights.append(parameter[k])
    if 0.0 in pooling_weights:
        # handle this case properly (discard it)
        pass
    else:
        # Kullback Leibler Divergence sum
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


def run(
    data_dir,
    area_of_interest,
    layers,
    pooling_weights,
    pooling_weights_auto,
    h3_resolution,
    save_normalized_layers,
    out_dir,
    normalization_quantiles=(0.0, 1.0),
):
    if len(layers) != len(pooling_weights):
        raise ValueError("layers and pooling_weights do not match.")

    hex_col = "h3_" + str(h3_resolution)
    gdf_h3_list = []
    z_list = []

    logging.info(f"Converting area of interest {Path(area_of_interest).name} to H3")
    if Path(area_of_interest).suffix in [".gpkg"]:
        gdf_aoi = geopandas.read_file(area_of_interest).to_crs(epsg=4326)
        gdf_aoi_h3 = (
            gdf_aoi.h3.polyfill_resample(resolution=h3_resolution, return_geometry=True).reset_index().to_crs(epsg=3857)
        )
        gdf_aoi_h3 = gdf_aoi_h3.rename(columns={"h3_polyfill": hex_col})
    else:
        raise NotImplementedError(f"layer_type {layer_file.suffix} not supported ['.gpkg']")

    for layer in layers:
        layer_file = next(Path(data_dir).rglob(layer))
        z = f"{layer_file.stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"
        if Path(Path(out_dir) / Path(f"{z}.gpkg")).exists():
            logging.info(f"Loading already existing H3 layer {z}.gpkg")
            gdf_h3 = geopandas.GeoDataFrame.from_file(Path(out_dir) / Path(f"{z}.gpkg")).to_crs(epsg=3857)
        else:
            logging.info(f"Converting layer {layer} to H3")
            if layer_file.suffix in [".tif", ".TIF", ".tiff", ".TIFF"]:
                # load raster to xyz dataframe
                df = (
                    rioxarray.open_rasterio(layer_file)
                    .rio.reproject("EPSG:4326")
                    .sel(band=1)
                    .to_pandas()
                    .stack()
                    .reset_index()
                    .rename(columns={"x": "lng", "y": "lat", 0: z})
                )
                # ignore background values
                df = df[df[z] != 0]

                # find hexs containing the points
                # raster values are summed for each h3 cell
                df[hex_col] = df.apply(lambda x: h3.geo_to_h3(x.lat, x.lng, h3_resolution), 1)
                df_h3 = df.groupby(hex_col)[z].sum().to_frame(z).reset_index()

                # build h3 geometry and subset columns
                geom = [Polygon(h3.h3_to_geo_boundary(row[hex_col], geo_json=True)) for index, row in df_h3.iterrows()]
                gdf_h3 = geopandas.GeoDataFrame(df_h3, geometry=geom, crs="EPSG:4326").to_crs(epsg=3857)
                gdf_h3 = gdf_h3[[hex_col, z, "geometry"]]

            elif layer_file.suffix in [".gpkg", ".geojson"]:
                # load vector to geodataframe
                gdf = geopandas.GeoDataFrame.from_file(layer_file, bbox=gdf_aoi_h3.to_crs(4326)).to_crs(epsg=4326)
                geom_type = gdf.geom_type[0]
                if geom_type == "Polygon":
                    # find hexs containing the polygons and build h3 geometry
                    # polygon values are assigned directly to the containing h3 cells
                    gdf_h3 = gdf.h3.polyfill_resample(resolution=h3_resolution, return_geometry=True).to_crs(epsg=3857)
                    index_name = gdf_h3.index.name
                    gdf_h3 = gdf_h3.reset_index().rename(columns={"value": z, index_name: hex_col})
                elif geom_type == "Point":
                    # find hexs containing the points and build h3 geometry
                    # point values are summed up to the containing h3 cells
                    gdf_h3 = gdf.h3.geo_to_h3_aggregate(
                        resolution=h3_resolution, operation="sum", return_geometry=True
                    ).to_crs(epsg=3857)
                    index_name = gdf_h3.index.name
                    gdf_h3 = gdf_h3.reset_index().rename(columns={"value": z, index_name: hex_col})
                else:
                    raise NotImplementedError(
                        f"geometry type of layer {layer_file.stem} not supported [Point, Polygon]"
                    )
                # subset columns
                gdf_h3 = gdf_h3[[hex_col, z, "geometry"]]

            else:
                raise NotImplementedError(
                    f"layer_type {layer_file.suffix} not supported ['.tif', '.TIF', '.tiff', '.TIFF', '.gpkg', '.geojson']"
                )

        logging.info(f"Joining layer to area of interest on H3 hex code")
        gdf_aoi_h3 = pd.merge(left=gdf_aoi_h3, right=gdf_h3, how="left", on=hex_col, suffixes=("", "_y"))
        gdf_aoi_h3.drop(gdf_aoi_h3.filter(regex="_y$").columns, axis=1, inplace=True)

        logging.info(f"Normalizing layer values (minmax scaler)")
        # normalize after joining and ensure that they are probability density functions, because the join modifies the spatial extent of the input layer and thus its statistics
        lo, hi = gdf_aoi_h3[z].quantile(normalization_quantiles[0]), gdf_aoi_h3[z].quantile(normalization_quantiles[1])
        gdf_aoi_h3[z] = np.clip((gdf_aoi_h3[z] - lo) / (hi - lo), a_min=0.0, a_max=1.0)

        # keep track of layer names
        z_list.append(z)

    # clean up joined layer
    z_list.insert(0, hex_col)
    z_list.append("geometry")
    gdf_h3 = gdf_aoi_h3[z_list]
    gdf_h3 = gdf_h3.fillna(0)

    logging.info(f"Probability pooling (loglinear)")
    # compose strings to specify normalization in file and column names
    norm = f"norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}"

    # get input layers
    p_layers = []
    for idx, layer in enumerate(layers):
        z = f"{Path(layer).stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"
        p_layers.append(gdf_h3[z].copy(deep=True))

    if pooling_weights_auto:
        logging.info("Searching optimal pooling weights")
        # compile parameter grid with pooling weights for each layer
        parameter_grid = {}
        start, stop, step = [0.0, 2.0, 0.25]
        for layer in layers:
            layer_name = Path(layer).stem
            parameter_grid[f"pooling_weights_{layer_name}"] = list(np.arange(start, stop + step, step))
        parameter_grid = list(ParameterGrid(parameter_grid))
        for i, p in enumerate(parameter_grid):
            p["id"] = i

        # compute hotspots for each pooling weight combination and return the KL divergence metric
        kldiv_list = Parallel(n_jobs=10)(
            delayed(kl_divergence_sum)(parameter=parameter, p_layers=p_layers) for parameter in tqdm(parameter_grid)
        )

        # compute optimal pooling weights (argmin(kl_sum))
        kl_sum_df = pd.concat(kldiv_list).sort_values("kl_sum", ascending=True)
        pooling_weights = kl_sum_df.iloc[0].values[0]
        logging.info(f"Using optimized pooling weights: {pooling_weights}")

    # compose strings to specify pooling weights in file and column names
    pw = "weights_"
    for pooling_weight in pooling_weights:
        pw += f"{str(pooling_weight).replace('.', '')}_"
    pw = pw[:-1]

    # pool layers
    gdf_h3[f"hotspots_{norm}_{pw}_{hex_col}"] = loglinear_pooling(weights=pooling_weights, p_layers=p_layers)

    logging.info("Exporting and plotting hotspots")
    hotspot_file = Path(out_dir) / Path(f"hotspots_{norm}_{pw}_{hex_col}.gpkg")
    gdf_h3.to_file(hotspot_file, driver="GPKG")
    bounds = gdf_h3.total_bounds
    plot_gdf_basemap(gdf_h3, column=f"hotspots_{norm}_{pw}_{hex_col}", cmap="Reds", bounds=bounds, add_labels=True)
    plt.savefig(Path(out_dir) / Path(f"hotspots_{norm}_{pw}_{hex_col}.png"), dpi=300)
    plt.close()

    if save_normalized_layers:
        logging.info("Exporting and plotting normalized h3 input data")
        for idx in range(len(layers)):
            z = f"{Path(layers[idx]).stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"
            gdf_h3.to_file(Path(out_dir) / Path(f"{z}.gpkg"), driver="GPKG")
            plot_gdf_basemap(gdf_h3, column=z, cmap="Reds", bounds=bounds, add_labels=True)
            plt.savefig(Path(out_dir) / Path(f"{z}.png"), dpi=300)
            plt.close()

    return hotspot_file
