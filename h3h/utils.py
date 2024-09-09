import contextily as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pyproj import Transformer


def plot_gdf_basemap(gdf, column, cmap="Reds", bounds=None, add_labels=True, center_cmap=False, crs=3857):
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
    # for provider in providers:
    #    print(provider)
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
    # ax.set_title(f"{column} ({gdf.shape[0]} hexagons)")
    ax.set_title(f"{column}")
    plt.tight_layout()


def regression_metrics(x, y):
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
