"""
Code to produce validation stats and graphs.

Jamie Taylor
Fariba Yousefi
2022-05-03
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
np.random.seed(42)

def predicted_vs_actual_ratio(pes_gsp_data):
    fig = plt.figure()
    plt.scatter(pes_gsp_data.loc[pes_gsp_data.set=="test", "gsp_pes_ratio"].to_numpy(),
                pes_gsp_data.loc[pes_gsp_data.set=="test", "predicted_GSP_PES_ratio"], s=0.01)
    plt.xlim(-0.1, 1)
    plt.ylim(-0.1, 1)
    plt.xlabel("Actual GSP:PES demand ratio")
    plt.ylabel("Predicted GSP:PES demand ratio")
    res = stats.linregress(
        pes_gsp_data.loc[pes_gsp_data.set=="test", "gsp_pes_ratio"].to_numpy(),
        pes_gsp_data.loc[pes_gsp_data.set=="test", "predicted_GSP_PES_ratio"].to_numpy()
    )
    fig.text(.7, .5, f"Linear fit: y = {res.slope:.2f}x + {res.intercept:.2f}\n"
             f"(r-squared: {res.rvalue**2:.3f})", ha='center')
    # plt.show()
    return

def plot_timeseries(pes_gsp_data, start="2018-07-01", end="2018-07-20", n_gsps=10):
    plt.rcParams['figure.dpi'] = 250
    ids = pes_gsp_data.loc[pes_gsp_data.set=="test", "region_id_20210423"].unique()
    for region_id in np.random.choice(ids, size=n_gsps, replace=False):
        plotdata = pes_gsp_data.loc[
            (pes_gsp_data.set=="test") & \
            (pes_gsp_data.region_id_20210423 == region_id) & \
            (pes_gsp_data.timestamp >= start) & \
            (pes_gsp_data.timestamp <= end)
        ].sort_values("timestamp")
        plt.figure()
        ax = plotdata.plot(x="timestamp", y="gsp_meter_volume", label="Actual", title=f"Region {region_id}");
        ax = plotdata.plot(x="timestamp", y="predicted_GSP_meter_volume", label="Predicted", ax=ax);
        plt.xlabel("Timestamp", fontsize=16)
        plt.ylabel("GSP demand (MW)", fontsize=16)
        plt.title(f"Region {region_id}", fontsize = 30)
#     plt.rcParams['figure.dpi'] = 150
    return

def histogram_residuals(pes_gsp_data):
    # import pdb; pdb.set_trace()
    residuals = pes_gsp_data.loc[pes_gsp_data.set=="test", "predicted_GSP_meter_volume"] - \
                pes_gsp_data.loc[pes_gsp_data.set=="test", "gsp_meter_volume"]
    norm_residuals = residuals / pes_gsp_data.loc[pes_gsp_data.set=="test", "gsp_meter_volume"]
    plt.figure()
    residuals[np.isfinite(residuals)].hist(bins=30)
    plt.figure()
    norm_residuals[np.isfinite(norm_residuals)].hist(bins=30)
    return

def run_validation(pes_gsp_data):
    predicted_vs_actual_ratio(pes_gsp_data)
    plot_timeseries(pes_gsp_data)
    histogram_residuals(pes_gsp_data)
    plt.show()