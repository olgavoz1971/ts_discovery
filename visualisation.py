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
    'a': {'before': 0.5, 'after': 13.5, 'color': '#80cbc4', 'alpha': 0.3},  # 14d
    'b': {'before': 1.0, 'after': 21.0, 'color': '#00897b', 'alpha': 0.5},  # 22d
    'c': {'before': 1.5, 'after': 23.5, 'color': '#004d40', 'alpha': 0.7}   # 25d
}
# To compare with a previous (symmetric-lc) approach
# ASYMMETRIC_CONFIG = {
#     'a': {'before': 7.0, 'after': 7.0, 'color': '#80cbc4', 'alpha': 0.3},  # 14d
#     'b': {'before': 11.0, 'after': 11.0, 'color': '#00897b', 'alpha': 0.5},  # 22d
#     'c': {'before': 12.5, 'after': 12.5, 'color': '#004d40', 'alpha': 0.7}  # 25d
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
    # T_obs - after <= T_peak <= T_obs + before
    start = obs_date - pd.Timedelta(days=cfg['after'])  # type: ignore
    end = obs_date + pd.Timedelta(days=cfg['before'])  # type: ignore
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


def run_refined_monte_carlo(df: pd.DataFrame, timespan_yrs, jitter_fraction=0.1, trials=1000):
    """
    Monte Carlo simulation with Asymmetric Light-Curve physics and Markovian jitter.
    """
    t0 = df['obs_date'].min()
    # Pre-process observations into relative days and asymmetric boundaries
    obs_list = []
    for _, row in df.iterrows():
        lim = str(row['limit']).strip().lower()
        cfg = ASYMMETRIC_CONFIG.get(lim, ASYMMETRIC_CONFIG['a'])
        day = (row['obs_date'] - t0).total_seconds() / 86400
        # obs_day - after <= current_t <= obs_day + before
        obs_list.append((day - cfg['after'], day + cfg['before']))

    total_days = timespan_yrs * 365.25
    periods_to_test = np.arange(1.1, 31.0, 0.05)  # Steps of 0.5yr for speed
    results = []

    for P in periods_to_test:
        hits = 0
        P_days = P * 365.25
        for _ in range(trials):
            current_t = np.random.uniform(0, P_days)
            detected = False
            while current_t < total_days:
                # Optimized detection check
                for start_bound, end_bound in obs_list:
                    if start_bound <= current_t <= end_bound:
                        detected = True
                        break
                if detected: break

                # Markov step with jitter (normal distribution centered at P)
                jitter = np.random.normal(0, jitter_fraction / 3) * P_days
                current_t += max(P_days + jitter, 1.0)

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
    fig, ax = plt.subplots(figsize=(20, 10))

    for _, row in df.iterrows():
        lim = str(row['limit']).strip().lower()
        cfg = ASYMMETRIC_CONFIG.get(lim, ASYMMETRIC_CONFIG['a'])

        # Asymmetric rectangle: height is total duration, but y-anchor is shifted
        # y_bottom = Center - cfg['after']
        total_duration = cfg['before'] + cfg['after']  # type: ignore
        rect = patches.Rectangle(
            (row['Season_Year'] - 0.4, row['Season_Day'] - cfg['after']),
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

    # 1. Create the legend elements
    legend_elements = [
        patches.Patch(color='#004d40', alpha=0.7, label='Limit C (25d)'),
        patches.Patch(color='#00897b', alpha=0.5, label='Limit B (22d)'),
        patches.Patch(color='#80cbc4', alpha=0.3, label='Limit A (14d)'),
        # The black line for the plate observation
        Line2D([0], [0], color='black', lw=1.5, label='Plate Midpoint')
    ]

    # Place legend BELOW the X-axis
    # loc='upper center' means the top-middle of the legend box
    # attaches to the coordinates (0.5, -0.15)
    ax.legend(handles=legend_elements,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),  # Adjust -0.12 to move further down if needed
              ncol=4,
              fontsize=16,
              # title="Plate Sensitivity and Outburst Duration",
              title_fontsize=18,
              frameon=False)

    # CRITICAL: Use subplots_adjust or tight_layout with rect
    # This prevents the legend from being "cut off" when saving the file
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
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


def take_instrument_readings(df: pd.DataFrame, target_conf=0.95, guessed_p=12.0):
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
    smooth_factor = 0.08
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


def main(file_path):
    df = load_archival_data(file_path)
    # 1. Geometry
    timespan_yrs, _, _ = calculate_refined_geometrical_completeness(df)
    # 2. Map
    plot_refined_seasonal_wrapped_coverage(df, season_start=SEASON_START_MD)
    # 3. Monte Carlo
    trials = 10000
    print(f"\nRunning Asymmetric MC Simulation ({trials} trials/step)...")
    mc_results = run_refined_monte_carlo(df, timespan_yrs=timespan_yrs, jitter_fraction=0.5, trials=trials)
    plot_detection_probability(mc_results)
    t_min, probability_at_guessed = take_instrument_readings(mc_results)
    print(f"period_min (95%)={t_min} {probability_at_guessed=}")


if __name__ == "__main__":
    path = '/home/voz/projects/UPJS/Shugarov/J0541/quiescent.dat'
    main(path)
