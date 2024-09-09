import logging
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from .utils import raster_to_h3, vector_to_h3, plot_gdf_basemap


def run(data_dir, layers, h3_resolution, out_dir, normalization_quantiles=(0.0, 1.0)):
    hex_col = "h3_" + str(h3_resolution)
    layer_files = []

    for layer in layers:
        logging.debug(f"Converting layer {layer} to H3")
        layer_file = next(Path(data_dir).rglob(layer))
        z = f"{layer_file.stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"
        if layer_file.suffix in [".tif", ".TIF", ".tiff", ".TIFF"]:
            # convert raster layer
            gdf_h3 = raster_to_h3(layer_file=layer_file, column_name=z, h3_resolution=h3_resolution)

        elif layer_file.suffix in [".gpkg", ".geojson"]:
            gdf_h3 = vector_to_h3(layer_file=layer_file, column_name=z, h3_resolution=h3_resolution, bbox=None)
        else:
            raise NotImplementedError(
                f"layer_type {layer_file.suffix} not supported ['.tif', '.TIF', '.tiff', '.TIFF', '.gpkg', '.geojson']"
            )

        logging.debug(f"Normalizing layer {layer}")
        lo, hi = gdf_h3[z].quantile(normalization_quantiles[0]), gdf_h3[z].quantile(normalization_quantiles[1])
        gdf_h3[z] = np.clip((gdf_h3[z] - lo) / (hi - lo), a_min=0.0, a_max=1.0)

        logging.debug(f"Exporting and plotting layer {layer} to file")
        gdf_h3.to_file(
            Path(out_dir) / Path(f"{z}.gpkg"),
            driver="GPKG",
        )
        plot_gdf_basemap(gdf_h3, column=z, cmap="Reds", bounds=gdf_h3.total_bounds)
        plt.savefig(
            Path(out_dir) / Path(f"{z}.png"),
            dpi=300,
        )
        plt.close()

        layer_files.append(Path(out_dir) / Path(f"{z}.gpkg"))

    return layer_files
