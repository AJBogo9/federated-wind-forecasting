"""Generate an interactive Plotly map of per-station test MSE for the presentation."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from data import build_node_datasets, fetch_station_names, fetch_wind_data, get_station_index
from graph import correlation_graph, geographic_graph

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_PATH = Path(__file__).parent.parent / "presentation" / "figures" / "station_mse_map.html"


def per_node_mse(weights: dict, datasets: dict, split: str) -> dict[str, float]:
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    result = {}
    for node, w in weights.items():
        if node not in datasets:
            continue
        X = datasets[node][x_key]
        y = datasets[node][y_key]
        residuals = X @ w - y
        result[node] = float(np.dot(residuals, residuals) / len(y))
    return result


def make_edge_coords(graph, station_ids: list[str], lats: list[float], lons: list[float]):
    """Return flat lat/lon lists for all edges, with None separators between segments."""
    idx = {n: i for i, n in enumerate(station_ids)}
    edge_lats: list = []
    edge_lons: list = []
    for i, j in graph.edge_index:
        if i >= len(station_ids) or j >= len(station_ids):
            continue
        ni, nj = station_ids[i], station_ids[j]
        if ni not in idx or nj not in idx:
            continue
        edge_lats += [lats[idx[ni]], lats[idx[nj]], None]
        edge_lons += [lons[idx[ni]], lons[idx[nj]], None]
    return edge_lats, edge_lons


def build_figure(
    nodes: list[str],
    lats: list[float],
    lons: list[float],
    all_mse: dict,
    station_names: dict,
    geo_graph,
    corr_graph,
) -> go.Figure:
    geo_edge_lats, geo_edge_lons = make_edge_coords(geo_graph, nodes, lats, lons)
    corr_edge_lats, corr_edge_lons = make_edge_coords(corr_graph, nodes, lats, lons)

    mse_all = [v for mse in all_mse.values() for v in mse.values()]
    cmin, cmax = min(mse_all), max(mse_all)

    geo_config = dict(
        marker_symbol="circle",
        edge_lat=geo_edge_lats,
        edge_lon=geo_edge_lons,
        edge_color="#e67e22",
    )
    corr_config = dict(
        marker_symbol="circle",
        edge_lat=corr_edge_lats,
        edge_lon=corr_edge_lons,
        edge_color="#1abc9c",
    )

    configs = {
        "Baseline": dict(marker_symbol="circle", edge_lat=[], edge_lon=[], edge_color="white"),
        "Geographic": geo_config,
        "Correlation": corr_config,
    }

    fig = go.Figure()

    # Trace 0, 1, 2: station markers for each system
    for label, cfg in configs.items():
        mse_vals = [all_mse[label].get(n, float("nan")) for n in nodes]
        fig.add_trace(go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers",
            name=label,
            visible=(label == "Baseline"),
            marker=dict(
                size=9,
                color=mse_vals,
                colorscale="RdYlGn_r",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Test MSE", thickness=12, x=1.0),
                symbol=cfg["marker_symbol"],
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"<b>{station_names.get(n, n)}</b><br>Test MSE: {all_mse[label].get(n, float('nan')):.4f}"
                for n in nodes
            ],
            hoverinfo="text",
            showlegend=False,
        ))

    # Trace 3, 4, 5: edge lines for each system (Baseline has no edges)
    for label, cfg in configs.items():
        fig.add_trace(go.Scattergeo(
            lat=cfg["edge_lat"],
            lon=cfg["edge_lon"],
            mode="lines",
            name=f"{label} edges",
            visible=(label == "Baseline"),
            line=dict(width=1, color=cfg["edge_color"]),
            hoverinfo="skip",
            showlegend=False,
        ))

    # One button per system: show matching station trace + edge trace, hide others
    # Trace order: [Baseline, Geo, Corr, Baseline-edges, Geo-edges, Corr-edges]
    buttons = []
    for i, label in enumerate(configs):
        visible = [False] * 6
        visible[i] = True      # station markers
        visible[i + 3] = True  # edges
        buttons.append(dict(label=label, method="update", args=[{"visible": visible}]))

    fig.update_layout(
        geo=dict(
            scope="europe",
            resolution=50,
            center=dict(lat=64.5, lon=25.5),
            projection_scale=5.5,
            showland=True, landcolor="#1a1a2e",
            showocean=True, oceancolor="#0d1117",
            showcoastlines=True, coastlinecolor="#444",
            showlakes=True, lakecolor="#0d1117",
            showcountries=True, countrycolor="#555",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5,
            xanchor="center",
            y=1.06,
            buttons=buttons,
            bgcolor="#1a1a2e",
            bordercolor="#555",
            font=dict(color="white"),
        )],
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )

    return fig


def main() -> None:
    print("Loading data...")
    df = fetch_wind_data()
    stations = get_station_index(df)
    datasets = build_node_datasets(df, stations)
    station_names = fetch_station_names(stations)
    geo_graph = geographic_graph(stations)
    corr_graph = correlation_graph(df, stations)

    systems = {"Baseline": "baseline", "Geographic": "geo", "Correlation": "corr"}
    all_mse: dict[str, dict[str, float]] = {}
    for label, fname in systems.items():
        data = np.load(RESULTS_DIR / f"{fname}.npz", allow_pickle=True)
        weights = data["weights"].item()
        all_mse[label] = per_node_mse(weights, datasets, "test")
        print(f"{label}: {len(all_mse[label])} stations")

    nodes = sorted(all_mse["Baseline"].keys())
    lats = [float(n.split("_")[0]) for n in nodes]
    lons = [float(n.split("_")[1]) for n in nodes]

    fig = build_figure(nodes, lats, lons, all_mse, station_names, geo_graph, corr_graph)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_PATH, include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
