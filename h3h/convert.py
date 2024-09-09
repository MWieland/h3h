import geopandas
import h3
import h3pandas
import logging
import matplotlib.pyplot as plt
import numpy as np
import rioxarray

from pathlib import Path
from shapely.geometry import Polygon
from .utils import plot_gdf_basemap


def run(data_dir, layers, h3_resolution, out_dir, normalization_quantiles=(0.0, 1.0)):
    hex_col = "h3_" + str(h3_resolution)
    layer_files = []

    for layer in layers:
        logging.debug(f"Converting layer {layer} to H3")
        layer_file = next(Path(data_dir).rglob(layer))
        z = f"{layer_file.stem}_norm_{str(normalization_quantiles[0]).replace('.', '')}_{str(normalization_quantiles[1]).replace('.', '')}_{hex_col}"

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
            df = df[df[z] == 1]

            # find hexs containing the points
            # raster values are summed for each h3 cell
            df[hex_col] = df.apply(lambda x: h3.geo_to_h3(x.lat, x.lng, h3_resolution), 1)
            df_h3 = df.groupby(hex_col)[z].sum().to_frame(z).reset_index()

            # normalize values (minmax scaler)
            lo, hi = df_h3[z].quantile(normalization_quantiles[0]), df_h3[z].quantile(normalization_quantiles[1])
            df_h3[z] = np.clip((df_h3[z] - lo) / (hi - lo), a_min=0.0, a_max=1.0)

            # build h3 geometry and subset columns
            geom = [Polygon(h3.h3_to_geo_boundary(row[hex_col], geo_json=True)) for index, row in df_h3.iterrows()]
            gdf_h3 = geopandas.GeoDataFrame(df_h3, geometry=geom, crs="EPSG:4326").to_crs(epsg=3857)
            gdf_h3 = gdf_h3[[hex_col, z, "geometry"]]

        elif layer_file.suffix in [".gpkg", ".geojson"]:
            # load vector to geodataframe
            gdf = geopandas.GeoDataFrame.from_file(layer_file).to_crs(epsg=4326)
            geom_type = gdf.geom_type[0]
            if geom_type == "Polygon":
                # find hexs containing the polygons and build h3 geometry
                # polygon values are assigned directly to the containing h3 cells
                gdf_h3 = (
                    gdf.h3.polyfill_resample(resolution=h3_resolution, return_geometry=True)
                    .reset_index()
                    .to_crs(epsg=3857)
                )
                gdf_h3 = gdf_h3.rename(columns={"h3_polyfill": hex_col})
            elif geom_type == "Point":
                # find hexs containing the points and build h3 geometry
                # point values are summed up to the containing h3 cells
                gdf_h3 = gdf.h3.geo_to_h3_aggregate(
                    resolution=h3_resolution, operation="sum", return_geometry=True
                ).to_crs(epsg=3857)
                index_name = gdf_h3.index.name
                gdf_h3 = gdf_h3.reset_index().rename(columns={index_name: hex_col})
            else:
                raise NotImplementedError(f"geometry type of layer {layer_file.stem} not supported [Point, Polygon]")

            # normalize values (minmax scaler)
            # NOTE: this requires an attribute named "value" in the input layer
            lo, hi = gdf_h3["value"].quantile(normalization_quantiles[0]), gdf_h3["value"].quantile(
                normalization_quantiles[1]
            )
            gdf_h3[z] = np.clip((gdf_h3["value"] - lo) / (hi - lo), a_min=0.0, a_max=1.0)

            # subset columns
            gdf_h3 = gdf_h3[[hex_col, z, "geometry"]]

        else:
            raise NotImplementedError(
                f"layer_type {layer_file.suffix} not supported ['.tif', '.TIF', '.tiff', '.TIFF', '.gpkg', '.geojson']"
            )

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
