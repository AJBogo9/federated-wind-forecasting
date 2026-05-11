"""FMI data fetching and feature engineering for wind speed forecasting.

Fetches hourly wind speed, temperature, relative humidity, and atmospheric
pressure from the FMI WFS API for full-year 2024. Builds per-node datasets
with lag features for federated learning.
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants — do NOT change; fixed for reproducibility
# ---------------------------------------------------------------------------

START: Final[str] = "2024-01-01"
END: Final[str] = "2024-12-31"
BBOX: Final[tuple[float, float, float, float]] = (19.0, 59.5, 31.6, 70.2)

# Lag window: use wind speed at t-1, t-2, t-3 as features
LAG: Final[int] = 3

# Random seed for weight initialisation
SEED: Final[int] = 42

# Minimum fraction of hourly timestamps that must be present
MIN_COMPLETENESS: Final[float] = 0.80

# Chronological split fractions
TRAIN_FRAC: Final[float] = 0.70
VAL_FRAC: Final[float] = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# FMI WFS API endpoint
_FMI_WFS_URL: Final[str] = "https://opendata.fmi.fi/wfs"
_STORED_QUERY: Final[str] = "fmi::observations::weather::hourly::simple"
_STATION_META_QUERY: Final[str] = "fmi::ef::stations"

# Variables to fetch (FMI parameter names)
_PARAMS: Final[str] = "ws_pt1h_avg,ta_pt1h_avg,rh_pt1h_avg,pa_pt1h_avg"

# Chunk size for API requests (one week avoids hitting the row limit)
_CHUNK_DAYS: Final[int] = 7

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class StationMeta(NamedTuple):
    """Metadata for a single FMI weather station."""

    name: str
    lat: float
    lon: float


class NodeDataset(TypedDict):
    """Per-node train/val/test tensors ready for FedGD.

    Feature matrix X has shape (n, 7):
        columns 0-2: ws_{t-3}, ws_{t-2}, ws_{t-1}  (lagged wind speed)
        column  3:   ta_{t-1}  (temperature one hour prior)
        column  4:   rh_{t-1}  (relative humidity one hour prior)
        column  5:   pa_{t-1}  (atmospheric pressure one hour prior)
        column  6:   1.0       (bias term)
    Label vector y has shape (n,): ws_{t+1}, z-score normalised.

    Both features and labels are z-score normalised using training-set
    statistics only. Normalising labels ensures that weight vectors live
    on a comparable scale across stations (different mean wind speeds),
    which is required for graph regularisation to function correctly.
    Use y_mean and y_std to convert predictions back to m/s.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # Normalisation stats (from training set) — stored for denormalisation
    feat_mean: np.ndarray  # shape (6,)
    feat_std: np.ndarray   # shape (6,)
    y_mean: float          # training-set mean of ws_{t+1}
    y_std: float           # training-set std of ws_{t+1}


# ---------------------------------------------------------------------------
# FMI API helpers
# ---------------------------------------------------------------------------


def _build_request_url(start_dt: datetime, end_dt: datetime) -> str:
    """Build a FMI WFS GetFeature URL for a single time chunk."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": _STORED_QUERY,
        "bbox": ",".join(str(v) for v in BBOX),
        "parameters": _PARAMS,
        "starttime": start_dt.strftime(fmt),
        "endtime": end_dt.strftime(fmt),
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{_FMI_WFS_URL}?{query}"


def _parse_fmi_xml(xml_text: str) -> pd.DataFrame:
    """Parse FMI WFS XML response into a tidy DataFrame.

    The FMI ``hourly::simple`` stored query returns BsWfsElement nodes with a
    gml:pos (lat lon) but no station name. We use the rounded coordinate pair
    as the station identifier, which is stable across requests.

    Returns a DataFrame with columns:
        station (str), lat (float), lon (float),
        time (datetime64[ns, UTC]), ws (float), ta (float), rh (float), pa (float)
    """
    import xml.etree.ElementTree as ET

    NS_BSWFS = "http://xml.fmi.fi/schema/wfs/2.0"
    NS_GML = "http://www.opengis.net/gml/3.2"

    root = ET.fromstring(xml_text)
    records: list[dict] = []

    for elem in root.iter(f"{{{NS_BSWFS}}}BsWfsElement"):
        location_el = elem.find(f"{{{NS_BSWFS}}}Location")
        time_el = elem.find(f"{{{NS_BSWFS}}}Time")
        param_el = elem.find(f"{{{NS_BSWFS}}}ParameterName")
        value_el = elem.find(f"{{{NS_BSWFS}}}ParameterValue")

        if None in (location_el, time_el, param_el, value_el):
            continue

        pos_el = location_el.find(f".//{{{NS_GML}}}pos")  # type: ignore[union-attr]
        if pos_el is None:
            continue

        pos_text = (pos_el.text or "").strip()
        parts = pos_text.split()
        if len(parts) < 2:
            continue

        lat = float(parts[0])
        lon = float(parts[1])

        # Use rounded coordinates as a stable station key
        station_id = f"{lat:.5f}_{lon:.5f}"

        param_name = (param_el.text or "").strip().lower()
        value_text = (value_el.text or "").strip()

        try:
            value = float(value_text)
        except ValueError:
            value = float("nan")

        records.append(
            {
                "station": station_id,
                "lat": lat,
                "lon": lon,
                "time": pd.Timestamp((time_el.text or "").strip()),
                "param": param_name,
                "value": value,
            }
        )

    if not records:
        return pd.DataFrame(columns=["station", "lat", "lon", "time", "ws", "ta", "rh", "pa"])

    df_long = pd.DataFrame(records)

    # Pivot from long to wide format
    param_map = {
        "ws_pt1h_avg": "ws",
        "ta_pt1h_avg": "ta",
        "rh_pt1h_avg": "rh",
        "pa_pt1h_avg": "pa",
    }
    df_long["param"] = df_long["param"].map(param_map)
    df_long = df_long.dropna(subset=["param"])

    # Keep lat/lon alongside for the pivot
    loc_df = df_long[["station", "lat", "lon"]].drop_duplicates("station")

    df_wide = df_long.pivot_table(
        index=["station", "time"], columns="param", values="value", aggfunc="first"
    ).reset_index()

    df_wide = df_wide.merge(loc_df, on="station", how="left")

    # Ensure all four columns exist
    for col in ("ws", "ta", "rh", "pa"):
        if col not in df_wide.columns:
            df_wide[col] = float("nan")

    return df_wide[["station", "lat", "lon", "time", "ws", "ta", "rh", "pa"]]


def _fetch_chunk(start_dt: datetime, end_dt: datetime, retries: int = 3) -> pd.DataFrame:
    """Fetch one time chunk from the FMI API with retry logic."""
    url = _build_request_url(start_dt, end_dt)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return _parse_fmi_xml(resp.text)
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"FMI API request failed for {start_dt.date()}–{end_dt.date()}: {exc}"
                ) from exc
            time.sleep(2 ** attempt)
    # unreachable
    return pd.DataFrame()


def _parse_station_names_xml(xml_text: str) -> dict[str, str]:
    """Parse FMI ef::stations XML into a coord-key -> name mapping.

    Matches on f"{lat:.5f}_{lon:.5f}" keys, the same format used by
    _parse_fmi_xml so the two dicts can be joined directly.
    """
    import xml.etree.ElementTree as ET

    NS_GML = "http://www.opengis.net/gml/3.2"
    NS_EF = "http://inspire.ec.europa.eu/schemas/ef/4.0"
    NS_SF = "http://inspire.ec.europa.eu/schemas/sf/4.0"

    root = ET.fromstring(xml_text)
    result: dict[str, str] = {}

    for facility in root.iter(f"{{{NS_EF}}}EnvironmentalMonitoringFacility"):
        name_el = facility.find(f"{{{NS_GML}}}name")
        pos_el = facility.find(
            f".//{{{NS_EF}}}representativePoint//{{{NS_GML}}}pos"
        )
        if name_el is None or pos_el is None:
            continue

        name = (name_el.text or "").strip()
        parts = (pos_el.text or "").strip().split()
        if len(parts) < 2 or not name:
            continue

        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            continue

        key = f"{lat:.5f}_{lon:.5f}"
        result[key] = name

    return result


def fetch_station_names(
    stations: pd.DataFrame,
    cache_path: Path = Path("data/station_names.json"),
    force: bool = False,
) -> dict[str, str]:
    """Return a mapping from coordinate station ID to human-readable FMI station name.

    Fetches the FMI ef::stations metadata endpoint once, matches each station
    by its rounded lat/lon key, and caches the result as JSON. Falls back to
    the coordinate ID string if a name cannot be resolved.

    Parameters
    ----------
    stations:
        Station index from ``get_station_index`` (index = coord station IDs).
    cache_path:
        JSON file to cache the name mapping.
    force:
        If True, re-fetch even when the cache exists.
    """
    import json

    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        with open(cache_path) as f:
            cached: dict[str, str] = json.load(f)
        # Return only stations we actually use, fill missing with coord ID
        return {s: cached.get(s, s) for s in stations.index}

    url = (
        f"{_FMI_WFS_URL}?service=WFS&version=2.0.0"
        f"&request=getFeature&storedquery_id={_STATION_META_QUERY}"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        name_map = _parse_station_names_xml(resp.text)
    except Exception as exc:
        print(f"Warning: could not fetch station names ({exc}). Using coord IDs.")
        return {s: s for s in stations.index}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(name_map, f, indent=2, ensure_ascii=False)

    matched = sum(1 for s in stations.index if s in name_map)
    print(f"Station names: {matched}/{len(stations.index)} matched from FMI metadata.")
    return {s: name_map.get(s, s) for s in stations.index}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_wind_data(
    start: str = START,
    end: str = END,
    cache_path: Path = Path("data/wind_2024.csv"),
    force: bool = False,
) -> pd.DataFrame:
    """Fetch hourly FMI wind/weather data and cache to CSV.

    Parameters
    ----------
    start:
        Start date string in YYYY-MM-DD format.
    end:
        End date (inclusive) in YYYY-MM-DD format.
    cache_path:
        Path to the CSV cache file.
    force:
        If True, re-fetch even when the cache exists.

    Returns
    -------
    DataFrame with columns: station, lat, lon, time, ws, ta, rh, pa.
    ``time`` is UTC-aware datetime64. Rows with NaN in ws are retained
    (they will be dropped during feature engineering).
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    print(f"Fetching FMI data {start} – {end} …")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end) + timedelta(days=1)  # inclusive end

    chunks: list[pd.DataFrame] = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS), end_dt)
        print(f"  {current.date()} → {chunk_end.date()} …", end=" ", flush=True)
        chunk = _fetch_chunk(current, chunk_end)
        print(f"{len(chunk)} rows")
        if not chunk.empty:
            chunks.append(chunk)
        current = chunk_end

    if not chunks:
        raise RuntimeError("No data returned from FMI API.")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["station", "time"])
    df = df.sort_values(["station", "time"]).reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} rows to {cache_path}")
    return df


def get_station_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of stations that meet the completeness threshold.

    Parameters
    ----------
    df:
        Raw DataFrame from ``fetch_wind_data``.

    Returns
    -------
    DataFrame indexed by station name with columns ``lat`` and ``lon``.
    Only stations with >= MIN_COMPLETENESS fraction of hours having a
    non-NaN wind speed observation are included.
    """
    # Total expected hourly timestamps in the dataset
    n_hours_total = df["time"].nunique()

    station_stats = (
        df.groupby("station")
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            ws_count=("ws", "count"),
        )
    )
    station_stats["completeness"] = station_stats["ws_count"] / n_hours_total

    qualified = station_stats[station_stats["completeness"] >= MIN_COMPLETENESS].copy()
    qualified = qualified.drop(columns=["ws_count", "completeness"])

    print(
        f"Station filter: {len(station_stats)} total → "
        f"{len(qualified)} with ≥{MIN_COMPLETENESS:.0%} completeness"
    )
    return qualified


def build_node_datasets(
    df: pd.DataFrame,
    stations: pd.DataFrame,
) -> dict[str, NodeDataset]:
    """Build per-node train/val/test datasets with lag features.

    Features (columns 0–5):
        ws_{t-3}, ws_{t-2}, ws_{t-1}   — lagged wind speed
        ta_t                            — temperature (°C)
        rh_t                            — relative humidity (%)
        pa_t                            — atmospheric pressure (hPa)
    Bias column (column 6): always 1.0.
    Label: ws_{t+1} — next-hour wind speed (m/s).

    The chronological split is 70% / 15% / 15%.
    Features are z-score normalised using training-set statistics only
    (bias column is not normalised).

    Parameters
    ----------
    df:
        Raw DataFrame from ``fetch_wind_data``.
    stations:
        Station index from ``get_station_index``.

    Returns
    -------
    Dict mapping station name → NodeDataset.
    """
    datasets: dict[str, NodeDataset] = {}

    for station_name in stations.index:
        sdf = (
            df[df["station"] == station_name]
            .sort_values("time")
            .reset_index(drop=True)[["time", "ws", "ta", "rh", "pa"]]
        )

        # Fill small gaps (≤2 hours) by linear interpolation, leave larger gaps as NaN
        sdf[["ws", "ta", "rh", "pa"]] = sdf[["ws", "ta", "rh", "pa"]].interpolate(
            method="linear", limit=2, limit_direction="forward"
        )

        # Build lag features
        for lag in range(1, LAG + 1):
            sdf[f"ws_lag{lag}"] = sdf["ws"].shift(lag)
        sdf["ta_lag1"] = sdf["ta"].shift(1)
        sdf["rh_lag1"] = sdf["rh"].shift(1)
        sdf["pa_lag1"] = sdf["pa"].shift(1)

        # Label: next-hour wind speed
        sdf["ws_next"] = sdf["ws"].shift(-1)

        # Drop rows with any NaN
        sdf = sdf.dropna().reset_index(drop=True)

        if len(sdf) < 50:
            # Not enough data after cleaning — skip this station
            continue

        # Feature matrix: [ws_{t-3}, ws_{t-2}, ws_{t-1}, ta_{t-1}, rh_{t-1}, pa_{t-1}]
        feature_cols = ["ws_lag3", "ws_lag2", "ws_lag1", "ta_lag1", "rh_lag1", "pa_lag1"]
        X_raw = sdf[feature_cols].to_numpy(dtype=np.float64)
        y = sdf["ws_next"].to_numpy(dtype=np.float64)

        # Chronological split
        n = len(sdf)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        X_train_raw = X_raw[:n_train]
        X_val_raw = X_raw[n_train : n_train + n_val]
        X_test_raw = X_raw[n_train + n_val :]
        y_train = y[:n_train]
        y_val = y[n_train : n_train + n_val]
        y_test = y[n_train + n_val :]

        # Normalise features using training-set statistics
        feat_mean = X_train_raw.mean(axis=0)
        feat_std = X_train_raw.std(axis=0)
        feat_std[feat_std == 0] = 1.0  # avoid division by zero

        def _norm(X: np.ndarray) -> np.ndarray:
            return (X - feat_mean) / feat_std

        # Add bias column
        def _with_bias(X: np.ndarray) -> np.ndarray:
            return np.hstack([_norm(X), np.ones((len(X), 1))])

        # Normalise labels using training-set statistics.
        # Without this, the bias weight encodes the station's mean wind speed
        # (e.g. ~7 m/s coastal vs ~3 m/s inland), making ||w_i - w_j||² large
        # even for meteorologically similar stations and preventing the graph
        # regularisation from functioning correctly.
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if y_std == 0.0:
            y_std = 1.0

        def _norm_y(y: np.ndarray) -> np.ndarray:
            return (y - y_mean) / y_std

        datasets[station_name] = NodeDataset(
            X_train=_with_bias(X_train_raw),
            y_train=_norm_y(y_train),
            X_val=_with_bias(X_val_raw),
            y_val=_norm_y(y_val),
            X_test=_with_bias(X_test_raw),
            y_test=_norm_y(y_test),
            feat_mean=feat_mean,
            feat_std=feat_std,
            y_mean=y_mean,
            y_std=y_std,
        )

    print(f"Built datasets for {len(datasets)} stations.")
    return datasets


def init_weights(
    datasets: dict[str, NodeDataset],
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Initialise per-node weight vectors from N(0, 0.01).

    Parameters
    ----------
    datasets:
        Per-node datasets from ``build_node_datasets``.
    seed:
        Random seed (fixed for reproducibility).

    Returns
    -------
    Dict mapping station name → weight vector of shape (7,).
    """
    rng = np.random.default_rng(seed)
    n_features = 7  # 6 features + bias
    return {
        station: rng.normal(0.0, 0.01, size=n_features)
        for station in datasets
    }
