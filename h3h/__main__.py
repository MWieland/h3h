import argparse
import datetime
import logging
import toml

from pathlib import Path

from . import convert, compare, gridsearch, hotspots


parser = argparse.ArgumentParser(
    description="""                                                                                                                              
    ██   ██ ██████  ██   ██ 
    ██   ██      ██ ██   ██ 
    ███████  █████  ███████ 
    ██   ██      ██ ██   ██ 
    ██   ██ ██████  ██   ██    

    h3h combines heterogeneous geospatial layers into the H3 grid system and computes hotspot map. 
""",
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument(
    "--settings",
    help=f"path to TOML file with settings.",
    required=True,
)

parser.add_argument(
    "--hotspots",
    action="store_true",
    help=f"compute hotspot map. input layers (raster, polygon, point) are converted to h3 grid and normalized on the fly.",
    required=False,
)

parser.add_argument(
    "--convert",
    action="store_true",
    help=f"convert layer (raster, polygon, point) to h3 grid and normalize.",
    required=False,
)

parser.add_argument(
    "--compare",
    action="store_true",
    help=f"compare two h3 layers for similarity.",
    required=False,
)

parser.add_argument(
    "--gridsearch",
    action="store_true",
    help=f"explore hyperparameter space using grid search.",
    required=False,
)

args = parser.parse_args()

if Path(args.settings).is_file():
    with open(args.settings) as f:
        settings = toml.load(f)
else:
    raise Exception("Cannot find settings TOML file.")

starttime = datetime.datetime.now()
logging.info(f"Starttime: {str(starttime.strftime('%H:%M:%S'))}")

if args.hotspots:
    logging.info("Computing hotspots")
    hotspot_file = hotspots.run(
        data_dir=settings["HOTSPOTS"]["DATA_DIR"],
        area_of_interest=settings["HOTSPOTS"]["AREA_OF_INTEREST"],
        layers=settings["HOTSPOTS"]["LAYERS"],
        pooling_weights=settings["HOTSPOTS"]["POOLING_WEIGHTS"],
        pooling_weights_auto=settings["HOTSPOTS"]["POOLING_WEIGHTS_AUTO"],
        h3_resolution=settings["HOTSPOTS"]["H3_RESOLUTION"],
        normalization_quantiles=settings["HOTSPOTS"]["NORMALIZATION_QUANTILES"],
        save_normalized_layers=settings["HOTSPOTS"]["SAVE_NORMALIZED_LAYERS"],
        out_dir=settings["HOTSPOTS"]["OUT_DIR"],
    )

if args.convert:
    logging.info("Converting layer to H3")
    layer_files = convert.run(
        data_dir=settings["CONVERT"]["DATA_DIR"],
        layers=settings["CONVERT"]["LAYERS"],
        normalization_quantiles=settings["CONVERT"]["NORMALIZATION_QUANTILES"],
        h3_resolution=settings["CONVERT"]["H3_RESOLUTION"],
        out_dir=settings["CONVERT"]["OUT_DIR"],
    )

if args.compare:
    logging.info("Comparing results")
    r_square_adj, r, p, aic, bic = compare.run(
        layer_true=settings["COMPARE"]["LAYER_TRUE"],
        layer_pred=settings["COMPARE"]["LAYER_PRED"],
        area_of_interest=settings["COMPARE"]["AREA_OF_INTEREST"],
        h3_resolution=settings["COMPARE"]["H3_RESOLUTION"],
        h3_resolution_aggregate=settings["COMPARE"]["H3_RESOLUTION_AGGREGATE"],
        h3_smoothing_factor=settings["COMPARE"]["H3_SMOOTHING_FACTOR"],
        intersect=settings["COMPARE"]["INTERSECT"],
        combine_plots=settings["COMPARE"]["COMBINE_PLOTS"],
        out_dir=settings["COMPARE"]["OUT_DIR"],
    )

if args.gridsearch:
    gridsearch.run(
        data_dir=settings["GRIDSEARCH"]["DATA_DIR"],
        layers=settings["GRIDSEARCH"]["LAYERS"],
        layer_true=settings["GRIDSEARCH"]["LAYER_TRUE"],
        area_of_interest=settings["GRIDSEARCH"]["AREA_OF_INTEREST"],
        pooling_weights_from_to_steps=settings["GRIDSEARCH"]["POOLING_WEIGHTS_FROM_TO_STEPS"],
        h3_resolutions=settings["GRIDSEARCH"]["H3_RESOLUTIONS"],
        normalization_quantiles=settings["GRIDSEARCH"]["NORMALIZATION_QUANTILES"],
        n_jobs=settings["GRIDSEARCH"]["N_JOBS"],
        save_geopackage=settings["GRIDSEARCH"]["SAVE_GEOPACKAGE"],
        out_dir=settings["GRIDSEARCH"]["OUT_DIR"],
    )

endtime = datetime.datetime.now()
logging.info(f"Endtime {str(endtime.strftime('%H:%M:%S'))}")
logging.info(f"{str((endtime - starttime).total_seconds())} sec")
