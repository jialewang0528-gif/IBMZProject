"""Build and execute a notebook that reads the live LinuxONE API, then export it as dark-theme HTML."""
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

out_dir = sys.argv[1]
cells = [
    nbformat.v4.new_markdown_cell("## CS2 Integrity Watch: demo map from the LinuxONE API"),
    nbformat.v4.new_code_cell(
        'import pandas as pd, requests\n'
        'API = "http://148.100.79.57:38888"\n'
        'requests.get(f"{API}/api/health").json()'),
    nbformat.v4.new_code_cell(
        'd = requests.get(f"{API}/api/series/2393088-2").json()\n'
        'df = pd.DataFrame(d["rounds"])\n'
        'cols = ["round_num", "score_team1", "score_team2", "buy_team1", "buy_team2",\n'
        '        "equip_diff_team1_minus_team2", "model_p", "market_p", "gap", "injected"]\n'
        'df.loc[10:16, cols]'),
    nbformat.v4.new_code_cell(
        'pd.DataFrame(requests.get(f"{API}/api/summary").json()["model_metrics"]).T[["brier", "logloss", "n"]]'),
    nbformat.v4.new_code_cell(
        'ax = df.plot(x="round_num", y=["model_p", "market_p"], marker="o", figsize=(9, 3),\n'
        '             color=["#1f77b4", "#ff7f0e"], grid=True)\n'
        'ax.axvspan(13.5, 16.5, color="#d62728", alpha=.12, label="injected prints")\n'
        'ax.set_ylabel("P(team1 wins map)"); ax.legend();'),
]
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=120, kernel_name="python3").execute()
nbformat.write(nb, f"{out_dir}/demo_map.ipynb")
exp = HTMLExporter(template_name="lab")
exp.theme = "dark"
body, _ = exp.from_notebook_node(nb)
open(f"{out_dir}/demo_map.html", "w").write(body)
print("ok")
