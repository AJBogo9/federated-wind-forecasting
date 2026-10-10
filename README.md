# Federated Wind Speed Forecasting in Finland

Course project for CS-E4740 Federated Learning at Aalto University, spring 2026.
The report is [`report/FLReport.pdf`](report/FLReport.pdf) (five pages).

Can weather stations improve each other's wind forecasts by sharing model parameters
instead of raw data? I tested this on hourly observations from 2024 from the Finnish
Meteorological Institute (FMI). Of the 165 stations with at least 80% of the year's hours
recorded, the 136 that also report temperature, humidity and pressure take part.

## Method

Each station is a node in a federated network and trains its own linear model. The model
predicts the wind speed at hour t+1 from the wind speeds at hours t−1, t−2 and t−3 and
the temperature, humidity and pressure at hour t−1. Features and labels are z-scored with
training-set statistics, and the data is split in time into training, validation and test
periods.

The stations are connected in a graph. Training minimizes each station's own error plus
a penalty that pulls the models of connected stations towards each other (GTVMin), and
runs as federated gradient descent (FedGD): each station uses only its own data and
the current parameters of its neighbours. A strength α sets how hard the penalty pulls,
and it is chosen on the validation period.

| System | Graph |
|---|---|
| Baseline | none: every station trains alone (α = 0) |
| Geographic | each station linked to its 5 nearest stations, weight falling with distance |
| Correlation | each station linked to the 5 stations whose wind speed correlates most with its own |

## Result

| System | α | Test MSE |
|---|---|---|
| Baseline | 0 | 0.2638 |
| Geographic | 0.1 | 0.2630 |
| Correlation | 0.1 | 0.2628 |

Mean over the 136 stations, on z-scored labels. The report has the full table and the
per-station breakdown.

Neither graph gives a practical improvement. The validation sweep picks a weak penalty,
and both graphs lower the test error by less than half a percent. Each station has a
full year of hourly data for a linear model with seven parameters, which leaves the
graph penalty little to improve. The report's conclusion lists where a graph could
matter: more flexible local models, and stations with much less data.

## Known issues

A review of the code after the course found three places where it differs from what the
report says. The code in this repository is left as graded.

| Issue | Where | What happens |
|---|---|---|
| Graph indexing | `src/run_experiments.py`, `src/fedgd.py` | Both graphs are built on all 165 stations, but FedGD looks up a station's row by its position among the 136 training stations. Of the station pairs the penalty couples, 23% (geographic) and 20% (correlation) are neighbours in the intended graph. The notebook and `src/make_station_mse_map.py` build the graphs the same way. |
| Forecast horizon | `src/data.py` | The features end at hour t−1 and the label is hour t+1, so the model forecasts two hours past its newest observation. The report calls it a next-hour forecast. |
| Gap filling | `src/data.py` | Linear interpolation with a two-hour limit fills the first two hours of every gap. The report says gaps longer than two hours are left missing. |

I reran the experiment with the adjacency re-indexed so that every training station uses
its own row. The sweep then picks a larger α, and the conclusion stays the same:

| System | α, as graded | Test MSE, as graded | α, re-indexed | Test MSE, re-indexed |
|---|---|---|---|---|
| Baseline | 0 | 0.2638 | 0 | 0.2638 |
| Geographic | 0.1 | 0.2630 | 0.5 | 0.2628 |
| Correlation | 0.1 | 0.2628 | 0.5 | 0.2627 |

## Repository

| Path | Contents |
|---|---|
| `src/data.py` | FMI download, station filter, features, time split |
| `src/graph.py` | the geographic and correlation graphs |
| `src/fedgd.py` | federated gradient descent |
| `src/run_experiments.py` | the α sweep and the three final runs |
| `src/generate_table.py` | the results table in the report |
| `src/make_station_mse_map.py` | per-station error map for the slides |
| `notebooks/report.qmd` | Quarto companion that draws every figure in the report |
| `report/` | LaTeX source and the submitted PDF |
| `presentation/` | slides (Quarto, reveal.js) |
| `peer-review-1/` | my response to the two peer reviewers |

## Running it

Needs [uv](https://docs.astral.sh/uv/). Quarto and LaTeX are only needed for the
figures and the PDF.

```bash
uv sync
uv run python src/run_experiments.py   # downloads the 2024 FMI data into data/, writes results/
uv run python src/generate_table.py    # writes report/results_table.tex
uv run quarto render notebooks/report.qmd   # writes figures/
cd report && latexmk -pdf -outdir=build FLReport.tex
```

## Reproducibility

The environment pins the package versions the project ran on. Two packages were missing
from the course's `requirements.txt` and are added: `jinja2` for the notebook's tables
and `pyyaml` for Quarto's Jupyter runner.

On 2026-10-10 I downloaded the 2024 data again and ran the steps above. The regenerated
`report/results_table.tex` is byte-identical to the submitted one, and the rebuilt
report matches the submitted text except for the access dates of two web references and
one figure label. Station names come from FMI's current station list, where 29 of the
stations no longer appear, so those stations are labelled by their coordinates.

## History

The course graded projects by double-blind peer review in two rounds, so the report PDF
carries no author name. The tag `final-submission` marks the project as submitted after
the peer-review revision. The commits after it add the uv environment, the PDF, this
README and the license, and leave the code unchanged.

## License

The code is under the MIT license. The style files in `report/` (`spconf.sty`,
`IEEEbib.bst`, `pdfplots.sty`, `ml_macros.tex`) come from the IEEE conference template
and the course template and keep their own terms.
