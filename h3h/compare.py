import geopandas
import h3
import h3pandas
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from PIL import Image
from .utils import regression_metrics, plot_gdf_basemap


def run(
    layer_true,
    layer_pred,
    area_of_interest,
    h3_resolution,
    out_dir,
    h3_resolution_aggregate=None,
    h3_smoothing_factor=0,
    intersect=False,
    combine_plots=False,
):
    hex_col = "h3_" + str(h3_resolution)
    h3_resolution_aggregate = None if h3_resolution_aggregate == "" else h3_resolution_aggregate

    logging.debug(f"Converting area of interest {Path(area_of_interest).name} to H3")
    if Path(area_of_interest).suffix in [".gpkg"]:
        gdf_aoi = geopandas.read_file(area_of_interest).to_crs(epsg=4326)
        gdf_aoi_h3 = (
            gdf_aoi.h3.polyfill_resample(resolution=h3_resolution, return_geometry=True).reset_index().to_crs(epsg=3857)
        )
        gdf_aoi_h3 = gdf_aoi_h3.rename(columns={"h3_polyfill": hex_col})
    else:
        raise NotImplementedError(f"layer_type {layer_file.suffix} not supported ['.gpkg']")

    z_list = []
    for layer_file in [layer_pred, layer_true]:
        layer_file = Path(layer_file)
        logging.debug(f"Loading H3 layer {layer_file.name}")
        gdf_layer_h3 = geopandas.read_file(layer_file).to_crs(epsg=3857)
        if hex_col not in list(gdf_layer_h3.columns.values):
            raise ValueError(
                f"cannot find h3 hex column {hex_col} in dataframe. is the provided layer in h3 grid and is the h3_resolution really {h3_resolution}?"
            )
        z_list.append(f"{layer_file.stem}")

        logging.debug(f"Joining layer to area of interest on H3 hex code")
        gdf_aoi_h3 = pd.merge(left=gdf_aoi_h3, right=gdf_layer_h3, how="left", on=hex_col, suffixes=("", "_y"))
        gdf_aoi_h3.drop(gdf_aoi_h3.filter(regex="_y$").columns, axis=1, inplace=True)
    gdf_h3 = gdf_aoi_h3[[hex_col, z_list[0], z_list[1], "geometry"]]

    if h3_resolution_aggregate is not None:
        logging.debug(f"Aggregating from H3 resolution {h3_resolution} to {h3_resolution_aggregate}")
        # this assumes h3 as index
        gdf_h3 = gdf_h3.set_index(hex_col).to_crs(epsg=4326)
        gdf_h3 = gdf_h3.h3.h3_to_parent_aggregate(
            resolution=h3_resolution_aggregate, operation="mean", return_geometry=True
        )
        h3_resolution = h3_resolution_aggregate
        hex_col_ = gdf_h3.index.name
        gdf_h3 = gdf_h3.reset_index().to_crs(epsg=3857)
        if h3_smoothing_factor > 0:
            # we also need to aggregate aoi to clip smoothing results at borders later
            gdf_aoi_h3 = gdf_aoi_h3[[hex_col, "geometry"]].set_index(hex_col).to_crs(epsg=4326)
            gdf_aoi_h3 = gdf_aoi_h3.h3.h3_to_parent_aggregate(
                resolution=h3_resolution_aggregate, operation="mean", return_geometry=True
            )
            gdf_aoi_h3 = gdf_aoi_h3.reset_index().to_crs(epsg=3857)
        hex_col = hex_col_
        aggr_str = f"_{hex_col}"
    else:
        aggr_str = ""

    if intersect:
        # compare only cells that spatially intersect between the input layers
        gdf_h3 = gdf_h3.dropna(how="any", subset=[z_list[0], z_list[1]])
    else:
        # compare all cells in the aoi
        # this fills the nan values in any input layer with 0 (=no hotspot)
        gdf_h3 = gdf_h3.fillna(0)

    if h3_smoothing_factor > 0:
        logging.debug(f"Smoothing H3 grid cells with factor k={h3_smoothing_factor}")
        # this assumes h3 as index
        gdf_h3 = gdf_h3.set_index(hex_col).to_crs(epsg=4326)
        gdf_h3 = gdf_h3.h3.k_ring_smoothing(h3_smoothing_factor, return_geometry=True)
        smooth_str = f"_smooth_{h3_smoothing_factor}"
        # smoothing adds cells at the borders
        # therefore clip smoothed cells to aoi
        gdf_h3 = gdf_h3.loc[gdf_aoi_h3[hex_col]]
        hex_col_ = gdf_h3.index.name
        gdf_h3 = gdf_h3.reset_index().to_crs(epsg=3857).rename(columns={hex_col_: hex_col})
    else:
        smooth_str = ""

    logging.debug("Comparing H3 layers inside area of interest")
    # normalize values (minmax scaler)
    # makes sure to have range [0,1] even after aggregation and/or smoothing
    normalization_quantiles = (0.0, 1.0)
    for i in range(len(z_list)):
        lo, hi = gdf_h3[z_list[i]].quantile(normalization_quantiles[0]), gdf_h3[z_list[i]].quantile(
            normalization_quantiles[1]
        )
        gdf_h3[z_list[i]] = round(np.clip((gdf_h3[z_list[i]] - lo) / (hi - lo), a_min=0.0, a_max=1.0), 3)

    # correlation analysis
    # x=independent variable (pred hotspots); y=dependent variable (true damages)
    r_square_adj, r, p, aic, bic, resid = regression_metrics(x=gdf_h3[z_list[0]], y=gdf_h3[z_list[1]])
    sns.jointplot(data=gdf_h3, x=z_list[0], y=z_list[1], kind="reg", xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    plt.title(f"h3_{h3_resolution}: adj R²={r_square_adj}, r={r}, p={p} \n aic={aic}, bic={bic}")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / Path(f"correlation_{z_list[0]}-{z_list[1]}{aggr_str}{smooth_str}.png"), dpi=300)
    plt.close()

    # plot predicted hotspot maps
    bounds = gdf_h3.total_bounds
    plot_gdf_basemap(gdf_h3, column=z_list[0], cmap="Reds", bounds=bounds, add_labels=False)
    plt.savefig(Path(out_dir) / Path(f"pred_{z_list[0]}{aggr_str}{smooth_str}.png"), dpi=300)
    plt.close()

    # plot true hotspot maps
    plot_gdf_basemap(gdf_h3, column=z_list[1], cmap="Reds", bounds=bounds, add_labels=False)
    plt.savefig(Path(out_dir) / Path(f"true_{z_list[1]}{aggr_str}{smooth_str}.png"), dpi=300)
    plt.close()

    # compare true and predicted hotspot maps by plotting the residuals
    gdf_h3 = pd.merge(left=gdf_h3, right=resid, how="left", left_index=True, right_index=True)
    plot_gdf_basemap(gdf_h3, column="residuals", cmap="seismic", bounds=bounds, add_labels=False, center_cmap=True)
    plt.savefig(Path(out_dir) / Path(f"residuals_{z_list[0]}-{z_list[1]}{aggr_str}{smooth_str}.png"), dpi=300)
    plt.close()

    # write comparison dataframe to file
    gdf_h3.to_file(
        Path(out_dir) / Path(f"comparison_{z_list[0]}-{z_list[1]}{aggr_str}{smooth_str}.gpkg"),
        driver="GPKG",
    )

    if combine_plots:
        logging.debug("Combining plots")
        img_files = [
            Path(out_dir) / Path(f"correlation_{z_list[0]}-{z_list[1]}{aggr_str}{smooth_str}.png"),
            Path(out_dir) / Path(f"pred_{z_list[0]}{aggr_str}{smooth_str}.png"),
            Path(out_dir) / Path(f"true_{z_list[1]}{aggr_str}{smooth_str}.png"),
            Path(out_dir) / Path(f"residuals_{z_list[0]}-{z_list[1]}{aggr_str}{smooth_str}.png"),
        ]
        imgs = [Image.open(x) for x in img_files]
        # resize all plots to plot with minimum height
        min_height = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1][1]
        imgs_resize = []
        for img in imgs:
            hpercent = min_height / float(img.size[1])
            wsize = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((wsize, min_height))
            imgs_resize.append(img)
        imgs_comb = Image.fromarray(np.hstack(imgs_resize))
        imgs_comb.save(Path(out_dir) / Path(f"comparison_{z_list[0]}-{z_list[1]}{aggr_str}{smooth_str}.png"))
        # keep only combined plot and true damage file
        for img_file in img_files:
            Path(img_file).unlink()

    return r_square_adj, r, p, aic, bic
