import geopandas
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from typing import Optional

from .utils import raster_to_h3, vector_to_h3, loglinear_pooling, kl_divergence_sum, plot_gdf_basemap


def run(
    data_dir: str,
    area_of_interest: str,
    layers: tuple[str, ...],
    pooling_weights: tuple[float, ...],
    pooling_weights_auto: bool,
    h3_resolution: int,
    save_normalized_layers: bool,
    out_dir: str,
    normalization_quantiles: Optional[tuple[float, ...]] = (0.0, 1.0),
) -> Path:
    if len(layers) != len(pooling_weights):
        raise ValueError("layers and pooling_weights do not match.")

    hex_col = "h3_" + str(h3_resolution)
    z_list = []

    logging.info(f"..Converting area of interest {Path(area_of_interest).name} to H3")
    if Path(area_of_interest).suffix in [".gpkg"]:
        gdf_aoi_h3 = vector_to_h3(layer_file=area_of_interest, column_name="aoi", h3_resolution=h3_resolution)
    else:
        raise NotImplementedError(f"layer_type {layer_file.suffix} not supported ['.gpkg']")

    for layer in layers:
        layer_file = next(Path(data_dir).rglob(layer))
        z = f"{layer_file.stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"
        if Path(Path(out_dir) / Path(f"{z}.gpkg")).exists():
            logging.info(f"..Loading already existing H3 layer {z}.gpkg")
            gdf_h3 = geopandas.GeoDataFrame.from_file(Path(out_dir) / Path(f"{z}.gpkg")).to_crs(epsg=3857)
        else:
            logging.info(f"..Converting layer {layer} to H3")
            if layer_file.suffix in [".tif", ".TIF", ".tiff", ".TIFF"]:
                # convert raster layer
                gdf_h3 = raster_to_h3(layer_file=layer_file, column_name=z, h3_resolution=h3_resolution)

            elif layer_file.suffix in [".gpkg", ".geojson"]:
                bbox = gdf_aoi_h3.to_crs(4326)
                gdf_h3 = vector_to_h3(layer_file=layer_file, column_name=z, h3_resolution=h3_resolution, bbox=bbox)
            else:
                raise NotImplementedError(
                    f"layer_type {layer_file.suffix} not supported ['.tif', '.TIF', '.tiff', '.TIFF', '.gpkg', '.geojson']"
                )

        logging.info(f"..Joining layer to area of interest on H3 hex code")
        gdf_aoi_h3 = pd.merge(left=gdf_aoi_h3, right=gdf_h3, how="left", on=hex_col, suffixes=("", "_y"))
        gdf_aoi_h3.drop(gdf_aoi_h3.filter(regex="_y$").columns, axis=1, inplace=True)

        logging.info(f"..Normalizing layer values (minmax scaler)")
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

    logging.info(f"..Probability pooling (loglinear)")
    # compose strings to specify normalization in file and column names
    norm = f"norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}"

    # get input layers
    p_layers = []
    for idx, layer in enumerate(layers):
        z = f"{Path(layer).stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"
        p_layers.append(gdf_h3[z].copy(deep=True))

    if pooling_weights_auto:
        logging.info("..Searching optimal pooling weights")
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
            delayed(kl_divergence_sum)(parameter=parameter, p_layers=p_layers)
            for parameter in tqdm(parameter_grid, desc="Searching optimal pooling weights")
        )

        # compute optimal pooling weights (argmin(kl_sum))
        kl_sum_df = pd.concat(kldiv_list).sort_values("kl_sum", ascending=True)
        pooling_weights = kl_sum_df.iloc[0].values[0]
        logging.info(f"..Using optimized pooling weights: {pooling_weights}")

    # compose strings to specify pooling weights in file and column names
    pw = "weights_"
    for pooling_weight in pooling_weights:
        pw += f"{str(pooling_weight).replace('.', '')}_"
    pw = pw[:-1]

    # pool layers
    gdf_h3[f"hotspots_{norm}_{pw}_{hex_col}"] = loglinear_pooling(weights=pooling_weights, p_layers=p_layers)

    logging.info("..Exporting and plotting hotspots")
    hotspot_file = Path(out_dir) / Path(f"hotspots_{norm}_{pw}_{hex_col}.gpkg")
    gdf_h3.to_file(hotspot_file, driver="GPKG")
    bounds = gdf_h3.total_bounds
    plot_gdf_basemap(gdf_h3, column=f"hotspots_{norm}_{pw}_{hex_col}", cmap="Reds", bounds=bounds, add_labels=True)
    plt.savefig(Path(out_dir) / Path(f"hotspots_{norm}_{pw}_{hex_col}.png"), dpi=300)
    plt.close()

    if save_normalized_layers:
        logging.info("..Exporting and plotting normalized h3 input data")
        for idx in range(len(layers)):
            z = f"{Path(layers[idx]).stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"
            gdf_h3.to_file(Path(out_dir) / Path(f"{z}.gpkg"), driver="GPKG")
            plot_gdf_basemap(gdf_h3, column=z, cmap="Reds", bounds=bounds, add_labels=True)
            plt.savefig(Path(out_dir) / Path(f"{z}.png"), dpi=300)
            plt.close()

    return hotspot_file
