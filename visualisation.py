import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from scipy.interpolate import make_interp_spline

# Global plot settings
plt.rcParams.update({'font.size': 20})

# --- TUNABLE PARAMETERS ---
SEASON_START_MD = (7, 15)  # (Month, Day) -> August 1st
GUESSED_PERIOD = 12  # Years between outbursts (if periodic)

# Updated Mapping: (Days_Visible_Before_Peak, Days_Visible_After_Peak)
# Based on 0.1 mag/day decline and steep rise observed in modern data.
ASYMMETRIC_CONFIG = {
    'a': {'rise': 0.5, 'fade': 13.5, 'color': '#80cbc4', 'alpha': 0.3, 'magB': 13.942},  # 14d
    'b': {'rise': 1.0, 'fade': 21.0, 'color': '#00897b', 'alpha': 0.5, 'magB': 15.610},  # 22d
    'c': {'rise': 1.5, 'fade': 23.5, 'color': '#004d40', 'alpha': 0.7, 'magB': 16.908}  # 25d
}


# To compare with a previous (symmetric-lc) approach
# ASYMMETRIC_CONFIG = {
#     'a': {'rise': 7.0, 'fade': 7.0, 'color': '#80cbc4', 'alpha': 0.3},  # 14d
#     'b': {'rise': 11.0, 'fade': 11.0, 'color': '#00897b', 'alpha': 0.5},  # 22d
#     'c': {'rise': 12.5, 'fade': 12.5, 'color': '#004d40', 'alpha': 0.7}  # 25d
# }


def get_asymmetric_window(obs_date, limit_key):
    """
    Helper to calculate the physical visibility window for a specific plate.
    Returns (start_of_outburst_window, end_of_outburst_window)
    """
    # cfg = ASYMMETRIC_CONFIG.get(str(limit_key).strip().lower(), ASYMMETRIC_CONFIG['a'])
    cfg = ASYMMETRIC_CONFIG.get(str(limit_key).strip().lower(), None)
    if cfg is None:
        raise ValueError(f"Unexpected limit {limit_key} at {obs_date=}")
    # If a plate is at T_obs, it can detect an outburst peaking at T_peak IF:
    # T_obs - fade <= T_peak <= T_obs + rise   (rise -- time interval before plates obs_time
    start = obs_date - pd.Timedelta(days=cfg['fade'])  # type: ignore
    end = obs_date + pd.Timedelta(days=cfg['rise'])  # type: ignore
    return start, end


def load_archival_data(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='|', skipinitialspace=True, engine='python', comment='#')
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['date_min'] = pd.to_datetime(df['date_min'].str.strip(), format='mixed')
    df['date_max'] = pd.to_datetime(df['date_max'].str.strip(), format='mixed')
    df['obs_date'] = df['date_min'] + (df['date_max'] - df['date_min']) / 2
    return df.sort_values('obs_date').reset_index(drop=True)


def calculate_refined_geometrical_completeness(df: pd.DataFrame):
    """
    Calculates completeness using asymmetric visibility windows.
    """
    diff = df['obs_date'].max() - df['obs_date'].min()
    timespan_days = diff.total_seconds() / 86400
    timespan_yrs = timespan_days / 365.25

    # 1. Generate asymmetric windows for every plate
    windows = [get_asymmetric_window(row['obs_date'], row['limit']) for _, row in df.iterrows()]
    windows.sort()

    # 2. Merge overlapping windows
    merged = []
    if windows:
        curr_start, curr_end = windows[0]
        for next_start, next_end in windows[1:]:
            if next_start <= curr_end:
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))

    protected_days = sum([(w[1] - w[0]).total_seconds() / 86400 for w in merged])
    completeness = (protected_days / timespan_days) * 100

    print(f"--- Asymmetric Completeness Report ---")
    print(f"Baseline: {timespan_yrs:.2f} yrs | Plates: {len(df)}")
    print(f"Effective Surveillance: {protected_days:.2f} days")
    print(f"Geometric Completeness: {completeness:.2f}%")
    return timespan_yrs, merged, completeness


def run_refined_monte_carlo(df: pd.DataFrame, timespan_yrs, period_step=0.05, jitter_fraction=0.1, trials=10000):
    """
    Monte Carlo simulation with Asymmetric Light-Curve physics and Markovian jitter.
    """
    t0 = df['obs_date'].min()
    # Pre-process observations into relative days and asymmetric boundaries
    obs_list = []
    for _, row in df.iterrows():
        limit_key = str(row['limit']).strip().lower()
        cfg = ASYMMETRIC_CONFIG.get(limit_key, None)
        if cfg is None:
            raise ValueError(f"Unexpected limit {limit_key} at {row['obs_date']}")
        day = (row['obs_date'] - t0).total_seconds() / 86400
        # obs_day - fade <= current_t <= obs_day + rise
        obs_list.append((day - cfg['fade'], day + cfg['rise']))

    total_days = timespan_yrs * 365.25
    period_start = 1.1
    period_stop = 31.1
    periods_to_test = np.arange(1.1, 31.0, period_step)
    results = []

    # set min_step to the longest outburst duration
    min_step = ASYMMETRIC_CONFIG['c']['fade'] + ASYMMETRIC_CONFIG['c']['rise']  # type: ignore
    for P in periods_to_test:
        print(f"{P=} ({(P-period_start)/(period_start - period_stop) * 100}%)")
        hits = 0
        P_days = P * 365.25
        for _ in range(trials):
            current_t = np.random.uniform(0, P_days)
            detected = False
            while current_t < total_days:
                # Optimised detection check
                for start_bound, end_bound in obs_list:  # Note: this is O(N_plates) but we are OK with this so far.
                    # In case of much larger collection we will consider correcting this logic to improve performance
                    if start_bound <= current_t <= end_bound:
                        detected = True
                        break
                if detected:
                    break

                # Step with jitter (normal distribution centred at P)
                jitter = np.random.normal(0, jitter_fraction / 3) * P_days
                current_t += max(P_days + jitter, min_step)

            if detected: hits += 1
        results.append({'period': P, 'prob_detection': hits / trials})

    return pd.DataFrame(results)


def plot_refined_seasonal_wrapped_coverage(df: pd.DataFrame, season_start):
    """
    Visualizes asymmetric windows on a 1-year wrapped seasonal grid.
    """

    def get_season_info(dt):
        sm, sd = season_start
        s_year = dt.year - 1 if (dt.month, dt.day) < (sm, sd) else dt.year
        return s_year, (dt - datetime(s_year, sm, sd)).days

    df['Season_Year'], df['Season_Day'] = zip(*df['obs_date'].apply(get_season_info))
    fig, ax = plt.subplots(figsize=(20, 13))

    for _, row in df.iterrows():
        lim = str(row['limit']).strip().lower()
        cfg = ASYMMETRIC_CONFIG.get(lim, None)
        if cfg is None:
            raise ValueError(f"plot_refined_seasonal_wrapped_coverage: wrong comparison star index {lim} "
                             f"at {row['obs_date']}")

        # Asymmetric rectangle: height is total duration, but y-anchor is shifted
        # y_bottom = Center - cfg['fade']
        total_duration = cfg['rise'] + cfg['fade']  # type: ignore
        rect = patches.Rectangle(
            (row['Season_Year'] - 0.4, row['Season_Day'] - cfg['fade']),
            0.8, total_duration,  # type: ignore
            color=cfg['color'], alpha=cfg['alpha'], zorder=2
        )
        ax.add_patch(rect)
        ax.hlines(row['Season_Day'], row['Season_Year'] - 0.4, row['Season_Year'] + 0.4,
                  colors='black', linewidth=0.8, zorder=3)

    # Styling
    ax.set_ylim(0, 365)
    ax.set_title("Historical Surveillance Map")
    ax.set_ylabel("Month of Observing Season")
    ax.set_xlabel("Season Start Year")

    # Custom Y-ticks (Monthly labels starting from August)
    month_names = []
    month_offsets = []
    curr = datetime(2000, season_start[0], season_start[1])
    for _ in range(12):
        month_offsets.append((curr - datetime(2000, season_start[0], season_start[1])).days)
        month_names.append(curr.strftime('%b'))
        curr = curr + pd.DateOffset(months=1)
    ax.set_yticks(month_offsets)
    ax.set_yticklabels(month_names)

    legend_elements = [
        patches.Patch(
            color=ASYMMETRIC_CONFIG[comp_star]['color'],  # type: ignore
            alpha=ASYMMETRIC_CONFIG[comp_star]['alpha'],
            label=(
                f'>{ASYMMETRIC_CONFIG[comp_star]["magB"]: .1f} mag'
                f'({ASYMMETRIC_CONFIG[comp_star]["rise"] + ASYMMETRIC_CONFIG[comp_star]["fade"]:.0f}d)'  # type: ignore
            )
        )
        for comp_star in ["a", "b", "c"]
    ]
    # The black line for the plate observation
    legend_elements += [Line2D([0], [0], color='black', lw=1.5, label='Plate Midpoint')]

    # Place legend BELOW the X-axis
    # loc='upper center' means the top-middle of the legend box
    # attaches to the coordinates (0.5, -0.15)
    ax.legend(  # region unfold
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # Adjust -0.12 to move further down if needed
        ncol=4,
        fontsize=16,
        # title="Plate Sensitivity and Outburst Duration",
        title_fontsize=18,
        frameon=False
    )  # endregion

    # CRITICAL: Use subplots_adjust or tight_layout with rect
    # This prevents the legend from being "cut off" when saving the file
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    plt.show()


# Set global publication styles
# Comment this at the working stage  - this make picture unbearable on my laptop
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,  # Standard size for papers
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "savefig.dpi": 300  # High resolution
})


def plot_refined_seasonal_wrapped_coverage_pub_version(df: pd.DataFrame, season_start):
    def get_season_info(dt):
        sm, sd = season_start
        s_year = dt.year - 1 if (dt.month, dt.day) < (sm, sd) else dt.year
        return s_year, (dt - datetime(s_year, sm, sd)).days

    df['Season_Year'], df['Season_Day'] = zip(*df['obs_date'].apply(get_season_info))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    # fig, ax = plt.subplots(figsize=(20, 10))

    for _, row in df.iterrows():
        lim = str(row['limit']).strip().lower()
        cfg = ASYMMETRIC_CONFIG.get(lim, ASYMMETRIC_CONFIG['a'])

        # Asymmetric rectangle: height is total duration, but y-anchor is shifted
        # y_bottom = Center - cfg['fade']
        total_duration = cfg['rise'] + cfg['fade']  # type: ignore
        rect = patches.Rectangle(
            (row['Season_Year'] - 0.4, row['Season_Day'] - cfg['fade']),
            0.8, total_duration,  # type: ignore
            color=cfg['color'], alpha=cfg['alpha'], zorder=2
        )
        ax.add_patch(rect)
        ax.hlines(row['Season_Day'], row['Season_Year'] - 0.4, row['Season_Year'] + 0.4,
                  colors='black', linewidth=0.8, zorder=3)

        # ax.hlines(row['Season_Day'], row['Season_Year'] - 0.4, row['Season_Year'] + 0.4,
        #           colors='black', linewidth=0.8, zorder=3)

    # Styling
    ax.set_ylim(0, 365)
    # ax.set_title("Historical Surveillance Map")
    ax.set_ylabel("Month of Observing Season")
    ax.set_xlabel("Season Start Year")

    # Custom Y-ticks (Monthly labels starting from August)
    month_names = []
    month_offsets = []
    curr = datetime(2000, season_start[0], season_start[1])
    for _ in range(12):
        month_offsets.append((curr - datetime(2000, season_start[0], season_start[1])).days)
        month_names.append(curr.strftime('%b'))
        curr = curr + pd.DateOffset(months=1)
    ax.set_yticks(month_offsets)
    ax.set_yticklabels(month_names)

    legend_elements = [
        patches.Patch(
            color=ASYMMETRIC_CONFIG[comp_star]['color'],  # type: ignore
            alpha=ASYMMETRIC_CONFIG[comp_star]['alpha'],
            label=(
                f'>{ASYMMETRIC_CONFIG[comp_star]["magB"]: .1f} mag'
                f'({ASYMMETRIC_CONFIG[comp_star]["rise"] + ASYMMETRIC_CONFIG[comp_star]["fade"]:.0f}d)'  # type: ignore
            )
        )
        for comp_star in ["a", "b", "c"]
    ]

    # Place legend BELOW the X-axis
    # loc='upper center' means the top-middle of the legend box
    # attaches to the coordinates (0.5, -0.15)
    ax.legend(  # region unfold
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),  # Adjust -0.12 to move further down if needed
        ncol=4,
        fontsize=16,
        frameon=False
    )  # endregion

    # CRITICAL: Use subplots_adjust or tight_layout with rect
    # This prevents the legend from being "cut off" when saving the file
    # plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    plt.tight_layout()
    # SAVE IN MULTIPLE FORMATS
    # PDF is best for the final LaTeX submission (vectorized)
    # PNG is good for quick previews or PPTs
    plt.savefig("time_coverage.pdf", bbox_inches='tight')
    plt.savefig("time_coverage.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_detection_probability(df: pd.DataFrame):
    df = df.sort_values('period')
    x, y = df['period'].to_numpy(), df['prob_detection'].to_numpy()
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = make_interp_spline(x, y, k=3)(x_smooth)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(x_smooth, y_smooth, linewidth=3, label='Asymmetric MC Simulation')
    ax.scatter(x, y, s=40, color='red', alpha=0.5)
    ax.set_xlabel("Recurrence Period [Years]")
    ax.set_ylabel("Probability of Historical Detection")
    ax.set_title("Survey Sensitivity Analysis")
    ax.grid(alpha=0.2)
    plt.show()


from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt


def take_instrument_readings(df: pd.DataFrame, target_conf=0.95,
                             guessed_p=12.0,
                             smooth_factor=0.08):
    """
    Fits a spline to MC results and 'reads' the instrument values.

    Parameters:
    df: DataFrame with 'period' and 'prob_detection'
    target_conf: The confidence level for the lower limit (e.g., 0.95)
    guessed_p: The period to check for detection probability (e.g., 12.0)
    """
    # 1. Prepare data
    df_sorted = df.sort_values('period')
    x = df_sorted['period'].values
    y = df_sorted['prob_detection'].values

    # 2. Fit a Smoothing Spline (s parameter controls smoothing)
    # Increase 's' if the curve is still too jumpy; decrease if it misses the trend

    spline = UnivariateSpline(x, y, s=smooth_factor)

    x_fine = np.linspace(x.min(), x.max(), 1000)
    y_fine = spline(x_fine)

    # 3. Find Lower Limit Period (T_min) where P_det >= target_conf
    # We find the first period where the spline stays BELOW the threshold
    # as we move from short to long periods.
    idx_limit = np.where(y_fine <= target_conf)[0]
    if len(idx_limit) > 0:
        t_min = x_fine[idx_limit[0]]
    else:
        t_min = x.min()

    # 4. Find Probability at the Guessed Period
    prob_at_guessed = spline(guessed_p)

    # --- VISUALIZATION ---
    plt.figure(figsize=(20, 10))
    plt.scatter(x, y, color='gray', alpha=0.3, label='Raw MC Trials (with Aliasing)')
    plt.plot(x_fine, y_fine, color='blue', linewidth=3, label='Spline Fit (Stochastic Trend)')

    # Reading Line: Lower Limit
    plt.axhline(target_conf, color='red', linestyle='--', alpha=0.6)
    plt.axvline(t_min, color='red', linestyle='--', alpha=0.6)
    plt.text(t_min + 0.5, target_conf + 0.02, f'T_min = {t_min:.2f} yrs', color='red', fontweight='bold')

    # Reading Line: Guessed Period
    plt.axvline(guessed_p, color='green', linestyle=':', alpha=0.8)
    plt.scatter([guessed_p], [prob_at_guessed], color='green', s=100, zorder=5)
    plt.text(guessed_p + 0.5, prob_at_guessed - 0.05, f'P({guessed_p}yr) = {prob_at_guessed:.2f}',
             color='green', fontweight='bold')

    plt.title(f"Instrument Reading: Detection Sensitivity (Conf: {target_conf * 100:.0f}%)", fontsize=18)
    plt.xlabel("Period [Years]", fontsize=16)
    plt.ylabel("Detection Probability", fontsize=16)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.2)
    plt.show()

    # Final "Scientific" Report
    print(f"--- Automated Instrument Readings ---")
    print(f"Lower Period Limit (at {target_conf * 100}% confidence): {t_min:.2f} years")
    print(f"Detection Probability for P={guessed_p} yrs       : {prob_at_guessed:.4f}")
    print("-" * 40)

    return t_min, prob_at_guessed


# I'm really crazy about metadata; let's handle it properly

def build_mc_metadata(df: pd.DataFrame, timespan_yrs, trials, period_step, jitter_fraction, mode):
    return {
        "n_rows": int(len(df)),
        # "t_min": float(df["time"].min()) if "time" in df else None,
        # "t_max": float(df["time"].max()) if "time" in df else None,
        "timespan_yrs": float(timespan_yrs),
        "trials": int(trials),
        "period_step": float(period_step),
        "jitter_fraction": float(jitter_fraction),
        "mode": mode,
    }


import pickle


def save_experiment(results_path, mc_results, metadata):
    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "mc_results": mc_results,
                "metadata": metadata
            },
            f
        )


def load_experiment(results_path):
    with open(results_path, "rb") as f:
        data = pickle.load(f)

    mc_results = data["mc_results"]
    metadata = data["metadata"]

    print("\n=== MC EXPERIMENT METADATA ===")
    for k, v in metadata.items():
        print(f"{k}: {v}")

    return mc_results, metadata


def main(file_path, mode="calculate", exp_path="mc_results.pkl"):
    df = load_archival_data(file_path)
    # Completeness
    timespan_yrs, _, _ = calculate_refined_geometrical_completeness(df)

    # Map
    plot_refined_seasonal_wrapped_coverage(df, season_start=SEASON_START_MD)
    plot_refined_seasonal_wrapped_coverage_pub_version(df, season_start=SEASON_START_MD)

    print(f"\nMonte Carlo mode: {mode}")
    if mode == "calculate":
        # Monte Carlo
        trials = 50000
        period_step = 0.02
        jitter_fraction = 0.5
        print(f"\nRunning Asymmetric MC Simulation ({trials} trials/step)...")
        mc_results = run_refined_monte_carlo(df, timespan_yrs=timespan_yrs,
                                             period_step=period_step,
                                             jitter_fraction=jitter_fraction,
                                             trials=trials)

        metadata = build_mc_metadata(
            df, timespan_yrs, trials, period_step, jitter_fraction, mode
        )
        # save to disk
        save_experiment(exp_path, mc_results, metadata)
        print(f"Saved experiment to {exp_path}")

    elif mode == "load":
        import os
        if not os.path.exists(exp_path):
            raise FileNotFoundError(
                f"MC cache not found: {exp_path}. "
                "Run with mode='calculate' first."
            )

        mc_results, metadata = load_experiment(exp_path)
        print(f"MC results loaded from: {exp_path}")
        print(f"{metadata=}")

    else:
        raise ValueError("mode must be 'calculate' or 'load'")

    plot_detection_probability(mc_results)
    target_conf = 0.95
    t_min, probability_at_guessed = take_instrument_readings(mc_results,
                                                             target_conf=target_conf,
                                                             guessed_p=12.0,
                                                             smooth_factor=0.01)
    print(f"period_min ({target_conf * 100}%)={t_min} {probability_at_guessed=}")

    target_conf = 0.5
    t_min, probability_at_guessed = take_instrument_readings(mc_results,
                                                             target_conf=target_conf,
                                                             guessed_p=12.0,
                                                             smooth_factor=0.01)
    print(f"period_min ({target_conf * 100}%)={t_min} {probability_at_guessed=}")


if __name__ == "__main__":
    path = '/home/voz/projects/UPJS/Shugarov/J0541/quiescent.dat'
    mode = 'load'
    # mode = 'calculate'
    main(path, mode=mode)
