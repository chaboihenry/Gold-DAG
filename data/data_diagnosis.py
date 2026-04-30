"""
Diagnostic script for gold_dag_data_combined.csv.
Inspects data quality and prints a structured console report.

Checks performed:
    1. Shape, date range, duplicate dates
    2. Weekend / non-business-day observations
    3. Per-variable missingness (pre-start, internal gaps, trailing)
    4. Suspicious values (negatives, zeros, infinities, extreme outliers)
    5. Business-day calendar coverage
    6. Cross-variable date alignment
"""

import pandas as pd
import numpy as np
from datetime import datetime

INPUT_CSV = "gold_dag_data_combined.csv"

# Variables that must always be positive (prices, indices)
POSITIVE_ONLY = ["eurusd", "vix", "dxy", "copper", "gvz", "wti"]
# Note: WTI did go negative on 2020-04-20 (the famous oil crash).

# Outlier threshold: daily log-returns above N sigma flag as suspicious
OUTLIER_SIGMA = 5.0


def header(text):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print('='*70)


def check_shape_and_dates(df):
    # Section 1: basic sanity on shape and date index
    header("1. SHAPE & DATE INDEX")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Range:   {df.index.min().date()} -> {df.index.max().date()}")

    # Duplicate dates would corrupt every downstream calculation
    n_dupes = df.index.duplicated().sum()
    flag = "OK" if n_dupes == 0 else "PROBLEM"
    print(f"Duplicate dates: {n_dupes}   [{flag}]")

    # Index sorted ascending?
    is_sorted = df.index.is_monotonic_increasing
    flag = "OK" if is_sorted else "PROBLEM"
    print(f"Sorted ascending: {is_sorted}   [{flag}]")


def check_weekends(df):
    # Section 2: weekend or other non-business-day observations
    header("2. WEEKEND / NON-BUSINESS DAY CHECK")
    # Monday=0 ... Sunday=6
    weekend_mask = df.index.dayofweek >= 5
    n_weekend = weekend_mask.sum()
    flag = "OK" if n_weekend == 0 else "REVIEW"
    print(f"Weekend rows: {n_weekend}   [{flag}]")
    if n_weekend > 0:
        print("Sample weekend dates:")
        for d in df.index[weekend_mask][:5]:
            print(f"  {d.date()}  ({d.strftime('%A')})")


def check_missingness(df):
    # Section 3: per-variable missingness, classified by region
    header("3. MISSINGNESS BY VARIABLE (CLASSIFIED)")
    print(f"{'col':8s}  {'first':12s}  {'last':12s}  "
          f"{'pre':>6s}  {'gaps':>6s}  {'trail':>6s}  {'total':>6s}")
    print("-" * 70)

    for col in df.columns:
        series = df[col]
        first = series.first_valid_index()
        last = series.last_valid_index()

        if first is None:
            print(f"{col:8s}  ALL MISSING")
            continue

        # Pre-start: NaN before the variable's first observation (expected)
        pre = series.loc[:first].isna().sum()
        # Trailing: NaN after the last observation (recent data not yet pub'd)
        trail = series.loc[last:].isna().sum()
        # Internal gaps: NaN strictly between first and last (the interesting one)
        active = series.loc[first:last]
        gaps = active.isna().sum()
        total = series.isna().sum()

        flag = "OK" if gaps == 0 else "REVIEW"
        print(f"{col:8s}  {first.date()}  {last.date()}  "
              f"{pre:>6d}  {gaps:>6d}  {trail:>6d}  {total:>6d}   [{flag}]")

    print("\nKey: 'gaps' = missing days inside the active range — these are")
    print("     the only ones that warrant investigation. 'pre' and 'trail'")
    print("     reflect series start dates and publication lag respectively.")


def check_internal_gaps_detail(df, max_show=10):
    # Section 3b: show actual dates where internal gaps occur
    header("3b. INTERNAL GAP DATES (sample)")

    for col in df.columns:
        series = df[col]
        first = series.first_valid_index()
        last = series.last_valid_index()
        if first is None:
            continue

        active = series.loc[first:last]
        # Restrict to business days inside the active window
        active_bdays = active.reindex(
            pd.bdate_range(first, last)
        )
        gap_dates = active_bdays[active_bdays.isna()].index

        if len(gap_dates) == 0:
            print(f"{col:8s}  no business-day gaps inside active range")
            continue

        print(f"{col:8s}  {len(gap_dates)} business-day gap(s); "
              f"showing first {min(max_show, len(gap_dates))}:")
        for d in gap_dates[:max_show]:
            print(f"            {d.date()}  ({d.strftime('%A')})")


def check_suspicious_values(df):
    # Section 4: negatives, zeros, infinities, extreme outliers
    header("4. SUSPICIOUS VALUES")

    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue

        n_inf = np.isinf(s).sum()
        n_zero = (s == 0).sum()
        n_neg = (s < 0).sum()

        # Daily log-returns; outliers detected on the return scale
        # (level outliers are meaningless because all these series have trends)
        rets = np.log(s.where(s > 0)).diff().dropna()
        if len(rets) > 0:
            sigma = rets.std()
            n_outliers = (rets.abs() > OUTLIER_SIGMA * sigma).sum()
        else:
            n_outliers = 0

        flags = []
        if n_inf > 0:
            flags.append(f"inf={n_inf}")
        if n_zero > 0:
            flags.append(f"zero={n_zero}")
        # WTI is the legitimate exception for negatives (2020-04-20 crash)
        if n_neg > 0 and col in POSITIVE_ONLY and col != "wti":
            flags.append(f"neg={n_neg}")
        elif n_neg > 0 and col == "wti":
            flags.append(f"neg={n_neg} (expected: 2020 oil crash)")

        outlier_flag = "REVIEW" if n_outliers > 0 else "OK"
        flag_str = "; ".join(flags) if flags else "OK"

        print(f"{col:8s}  min={s.min():>10.4f}  max={s.max():>10.4f}  "
              f"{flag_str}")
        print(f"          {OUTLIER_SIGMA:.0f}-sigma return outliers: "
              f"{n_outliers}   [{outlier_flag}]")


def check_calendar_coverage(df):
    # Section 5: % of US business days covered per variable
    header("5. BUSINESS-DAY CALENDAR COVERAGE")
    print(f"{'col':8s}  {'active range':25s}  {'b-days':>7s}  "
          f"{'observed':>9s}  {'coverage':>9s}")
    print("-" * 70)

    for col in df.columns:
        s = df[col]
        first = s.first_valid_index()
        last = s.last_valid_index()
        if first is None:
            continue

        # Total US business days in the active window
        bdays = pd.bdate_range(first, last)
        n_bdays = len(bdays)
        n_obs = s.loc[first:last].notna().sum()
        cov = 100.0 * n_obs / n_bdays if n_bdays else 0.0

        # >97% expected (allows for federal holidays not on bday calendar)
        flag = "OK" if cov >= 97.0 else "REVIEW"
        print(f"{col:8s}  {first.date()} -> {last.date()}  "
              f"{n_bdays:>7d}  {n_obs:>9d}  {cov:>7.2f}%   [{flag}]")


def check_cross_alignment(df):
    # Section 6: how often do all variables have data on the same day?
    header("6. CROSS-VARIABLE DATE ALIGNMENT")

    # Restrict to dates where the most-recent-starting variable is available
    latest_start = max(df[c].first_valid_index() for c in df.columns
                       if df[c].first_valid_index() is not None)
    common_window = df.loc[latest_start:]

    n_total = len(common_window)
    n_complete = common_window.dropna().shape[0]
    n_anymissing = n_total - n_complete

    print(f"Common window starts: {latest_start.date()} "
          f"(latest first-valid date across all columns)")
    print(f"Rows in common window:  {n_total:,}")
    print(f"Rows with ALL columns:  {n_complete:,}")
    print(f"Rows with ANY missing:  {n_anymissing:,}  "
          f"({100*n_anymissing/n_total:.1f}%)")

    print("\nMissingness breakdown by # of missing columns in common window:")
    miss_counts = common_window.isna().sum(axis=1).value_counts().sort_index()
    for n_missing, count in miss_counts.items():
        print(f"  {n_missing} cols missing: {count:>6d} rows")


def main():
    print(f"\nDiagnostic report for: {INPUT_CSV}")
    print(f"Generated: {datetime.now().isoformat(timespec='seconds')}")

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"], index_col="date")

    check_shape_and_dates(df)
    check_weekends(df)
    check_missingness(df)
    check_internal_gaps_detail(df)
    check_suspicious_values(df)
    check_calendar_coverage(df)
    check_cross_alignment(df)

    print("\n" + "=" * 70)
    print("  END OF REPORT")
    print("=" * 70)
    print("\nInterpretation guide:")
    print("  [OK]      - check passed")
    print("  [REVIEW]  - investigate before handoff; may or may not be a problem")
    print("  [PROBLEM] - definite issue; do not hand off until resolved")


if __name__ == "__main__":
    main()