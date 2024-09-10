import logging
import numpy as np
import pandas as pd
import shutil
import tempfile

from joblib import Parallel, delayed
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from .hotspots import run as hotspots_run
from .compare import run as compare_run
from .convert import run as convert_run


def compare(
    parameter: dict, data_dir: str, layers: tuple[str, ...], layer_true: str, area_of_interest: str, out_dir: str
) -> dict:
    pooling_weights = []
    for k in parameter.keys():
        if "pooling_weights" in k:
            pooling_weights.append(parameter[k])
    h3_resolution = parameter["h3_resolution"]
    normalization_quantiles = parameter["normalization_quantiles"]
    parameter_id = parameter["id"]

    if sum(pooling_weights) == 0.0:
        # handle this case properly (discard it)
        r_square_adj, r, p, aic, bic, file_name = None, None, None, None, None, None

    else:
        try:
            # compute hotspot map
            layer_pred = hotspots_run(
                data_dir=data_dir,
                area_of_interest=area_of_interest,
                layers=layers,
                pooling_weights=pooling_weights,
                pooling_weights_auto=False,
                h3_resolution=h3_resolution,
                normalization_quantiles=normalization_quantiles,
                save_normalized_layers=False,
                out_dir=out_dir,
            )

            with tempfile.TemporaryDirectory(prefix=f"id_{parameter_id}_") as tmp_dir:
                # convert true layer to h3 for given resolution
                # set normalization quantiles always to [0, 1] to be consistent with hotspots (prediction) normalization
                # in hotspots we normalize input layers with user defined quantiles, but the actual hotspots
                # are always normalized with [0, 1] in the final step
                layer_files = convert_run(
                    data_dir=data_dir,
                    layers=[layer_true],
                    h3_resolution=h3_resolution,
                    out_dir=tmp_dir,
                    normalization_quantiles=[0.0, 1.0],
                )

                # compare true and predicted hotspot map
                # using tmp_dir here cause of all the plotting in compare.py
                r_square_adj, r, p, aic, bic = compare_run(
                    layer_true=layer_files[0],
                    layer_pred=layer_pred,
                    area_of_interest=area_of_interest,
                    h3_resolution=h3_resolution,
                    combine_plots=True,
                    out_dir=tmp_dir,
                )
                for f in Path(tmp_dir).glob("comparison_*"):
                    shutil.copyfile(f, Path(out_dir) / Path(f"comparisons_h3_{h3_resolution}") / Path(f).name)
                file_name = Path(f).stem
        except Exception as e:
            logging.debug(e)
            r_square_adj, r, p, aic, bic, file_name = None, None, None, None, None, None

    return {
        "parameter_id": parameter_id,
        "h3_resolution": h3_resolution,
        "normalization_quantiles": normalization_quantiles,
        "pooling_weights": pooling_weights,
        "layers": layers,
        "r_square_adj": r_square_adj,
        "r": r,
        "p": p,
        "aic": aic,
        "bic": bic,
        "file_name": file_name,
    }


def run(
    data_dir: str,
    layers: tuple[str, ...],
    layer_true: str,
    area_of_interest: str,
    pooling_weights_from_to_steps: tuple[float, float, float],
    h3_resolutions: int,
    normalization_quantiles: tuple[float, float],
    n_jobs: int,
    save_geopackage: bool,
    out_dir: str,
) -> pd.DataFrame:
    for h3_resolution in h3_resolutions:
        logging.info(f"Grid search hyperparameters for h3_resolution {h3_resolution}")

        # make output subdirectories for plots
        Path(Path(out_dir) / Path(f"hotspots_h3_{h3_resolution}")).mkdir(parents=True, exist_ok=True)
        Path(Path(out_dir) / Path(f"comparisons_h3_{h3_resolution}")).mkdir(parents=True, exist_ok=True)

        # compile parameter grid with pooling weights for each layer
        parameter_grid = {
            "h3_resolution": [h3_resolution],
            "normalization_quantiles": normalization_quantiles,
        }
        start, stop, step = pooling_weights_from_to_steps
        for layer in layers:
            layer_name = Path(layer).stem
            parameter_grid[f"pooling_weights_{layer_name}"] = list(np.arange(start, stop + step, step))
        parameter_grid = list(ParameterGrid(parameter_grid))
        for i, p in enumerate(parameter_grid):
            p["id"] = i

        # run gridsearch in multiprocessing
        processed_list = Parallel(n_jobs=n_jobs)(
            delayed(compare)(parameter, data_dir, layers, layer_true, area_of_interest, out_dir)
            for parameter in tqdm(parameter_grid)
        )

        # copy hotspot files to subdirectory
        for f in Path(out_dir).glob("hotspots_*.*"):
            shutil.move(f, Path(out_dir) / Path(f"hotspots_h3_{h3_resolution}") / Path(f).name)

        if save_geopackage is False:
            # remove all geopackage files from results
            for f in Path(Path(out_dir) / Path(f"hotspots_h3_{h3_resolution}")).glob("hotspots_*.gpkg"):
                f.unlink()
            for f in Path(Path(out_dir) / Path(f"comparisons_h3_{h3_resolution}")).glob("comparison_*.gpkg"):
                f.unlink()

        # save results to file
        results = pd.DataFrame(processed_list)
        results.to_csv(Path(out_dir) / Path(f"comparison_results_h3_{h3_resolution}.csv"), sep=";")

        # report top5 parameter combinations (having highest r_squared_adj)
        top5 = results.sort_values(by=["r_square_adj"], ascending=False)[0:5]
        logging.info(top5)

        return results
