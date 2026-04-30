"""
Fetch daily max-history data for 6 gold-DAG variables.
FRED is the primary source; Yahoo Finance is used only when FRED has no
equivalent series.

Variables:
    eurusd  - EUR/USD spot rate                 (FRED: DEXUSEU)
    wti     - WTI crude oil spot                (FRED: DCOILWTICO)
    vix     - CBOE Volatility Index             (FRED: VIXCLS)
    dxy     - ICE US Dollar Index               (Yahoo: DX-Y.NYB) - not on FRED
    copper  - COMEX copper front-month futures  (Yahoo: HG=F)     - FRED daily n/a
    gvz     - CBOE Gold Volatility Index        (Yahoo: ^GVZ)     - not on FRED
"""

import io
import subprocess
import pandas as pd
import yfinance as yf
from datetime import datetime

# Output paths
OUTPUT_CSV = "gold_dag_data_combined.csv"
SOURCES_TXT = "gold_dag_data_sources.txt"

# Pull max history; FRED returns full series start regardless of START
START = "1985-01-01"
END = datetime.today().strftime("%Y-%m-%d")

# FRED series: column_name -> (series_id, description, native_start)
fred_series = {
    "eurusd": ("DEXUSEU",    "EUR/USD Spot Rate (USD per EUR)",     "1999-01-04"),
    "wti":    ("DCOILWTICO", "WTI Crude Oil Spot Price (USD/bbl)",  "1986-01-02"),
    "vix":    ("VIXCLS",     "CBOE Volatility Index (close)",       "1990-01-02"),
}

# Yahoo tickers used only because FRED has no equivalent
yahoo_series = {
    "dxy":    ("DX-Y.NYB", "ICE US Dollar Index",                       "1985-01-01"),
    "copper": ("HG=F",     "COMEX Copper Front-Month Futures (USD/lb)", "2000-08-23"),
    "gvz":    ("^GVZ",     "CBOE Gold Volatility Index",                "2008-06-03"),
}


def fetch_fred_one(sid, timeout=60):
    # Fetch a single FRED series via system curl. Returns CSV text.
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    # -s silent, -S show errors, --fail nonzero exit on HTTP error
    result = subprocess.run(
        ["curl", "-sS", "--fail", url],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr.strip()}")
    return result.stdout


def fetch_fred(series_map, start, end):
    # Pull FRED series via curl. Returns combined dataframe.
    frames = []
    for col, (sid, _desc, _start) in series_map.items():
        print(f"  FRED:  {sid:12s} -> {col}")
        try:
            csv_text = fetch_fred_one(sid)
            s = pd.read_csv(io.StringIO(csv_text),
                            parse_dates=["observation_date"],
                            index_col="observation_date",
                            na_values=["."])
            s.columns = [col]
            s = s.loc[start:end]
            frames.append(s)
        except Exception as e:
            print(f"    ! failed: {e}")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_yahoo(ticker_map, start, end):
    # Pull Yahoo Finance close prices. Returns combined dataframe.
    frames = []
    for col, (tkr, _desc, _start) in ticker_map.items():
        print(f"  Yahoo: {tkr:12s} -> {col}")
        try:
            # auto_adjust=True applies splits/dividends; matters for ETFs
            df = yf.download(tkr, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"    ! empty result")
                continue
            # Handle MultiIndex columns from newer yfinance versions
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = col
            frames.append(close.to_frame())
        except Exception as e:
            print(f"    ! failed: {e}")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def write_sources(path, fred_map, yahoo_map):
    # Write a plain-text source-attribution file.
    with open(path, "w") as f:
        f.write("GOLD DAG DATA - SOURCES\n")
        f.write(f"Generated: {datetime.today().isoformat()}\n")
        f.write("Frequency: Daily (business days)\n\n")

        f.write("=== Primary source: FRED ===\n")
        f.write("https://fred.stlouisfed.org\n\n")
        for col, (sid, desc, native) in fred_map.items():
            url = f"https://fred.stlouisfed.org/series/{sid}"
            f.write(f"{col:8s} | {sid:12s} | starts {native} | {desc}\n")
            f.write(f"{'':8s}   {url}\n")

        f.write("\n=== Fallback source: Yahoo Finance ===\n")
        f.write("https://finance.yahoo.com\n")
        f.write("Used only for variables not hosted on FRED.\n")
        f.write("Field: Close (split/dividend-adjusted via auto_adjust=True)\n\n")
        for col, (tkr, desc, native) in yahoo_map.items():
            url = f"https://finance.yahoo.com/quote/{tkr}"
            f.write(f"{col:8s} | {tkr:12s} | starts {native} | {desc}\n")
            f.write(f"{'':8s}   {url}\n")


def main():
    print(f"Date range: {START} -> {END}\n")

    print("Fetching FRED series (primary, via curl)...")
    df_fred = fetch_fred(fred_series, START, END)

    print("\nFetching Yahoo series (fallback for non-FRED variables)...")
    df_yahoo = fetch_yahoo(yahoo_series, START, END)

    if df_fred.empty and df_yahoo.empty:
        print("\nNo data fetched. Aborting.")
        return

    # Outer join preserves all dates across both sources
    df_all = pd.concat([df_fred, df_yahoo], axis=1).sort_index()
    df_all.index.name = "date"
    df_all = df_all.dropna(how="all")

    # Reorder columns: FRED-sourced first, Yahoo second
    col_order = list(fred_series.keys()) + list(yahoo_series.keys())
    df_all = df_all[[c for c in col_order if c in df_all.columns]]

    df_all.to_csv(OUTPUT_CSV)
    write_sources(SOURCES_TXT, fred_series, yahoo_series)

    print(f"\nWrote {OUTPUT_CSV}")
    print(f"Wrote {SOURCES_TXT}")
    print(f"\nShape: {df_all.shape}")
    print(f"Range: {df_all.index.min().date()} -> {df_all.index.max().date()}")
    print("\nFirst-valid date per column:")
    for c in df_all.columns:
        first = df_all[c].first_valid_index()
        last = df_all[c].last_valid_index()
        nobs = df_all[c].notna().sum()
        print(f"  {c:8s} {first.date() if first else 'NA'} -> "
              f"{last.date() if last else 'NA'}  (n={nobs})")


if __name__ == "__main__":
    main()