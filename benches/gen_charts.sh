#!/bin/bash

python3 << 'PYTHON_SCRIPT'
import json
import urllib.parse

BASE_64 = "../target/criterion/Maps Benchmark in 0~64 Scale"
BASE_196 = "../target/criterion/Maps Benchmark in 0~196 Scale"

IMPLS = {
    "small-map (FxHash)": ("small-map__FxSmallMap ⬅", "#4CAF50"),
    "FxHashMap": ("rustc_hash__FxHashMap", "#2196F3"),
    "micromap": ("micromap__Map", "#FF9800"),
    "hashbrown": ("hashbrown__HashMap", "#9C27B0"),
    "std HashMap": ("std__collections__HashMap", "#607D8B"),
    "BTreeMap": ("std__collections__BTreeMap", "#795548"),
    "litemap": ("litemap__LiteMap", "#E91E63"),
    "tinymap": ("tinymap__array_map__ArrayMap", "#00BCD4"),
}

def get_time(base_dir, impl_dir, size):
    path = f"{base_dir}/{impl_dir}/{size}/new/estimates.json"
    try:
        with open(path) as f:
            data = json.load(f)
            return round(data["mean"]["point_estimate"], 2)
    except:
        return None

def make_chart_url(title, sizes, base_dir, impl_names):
    datasets = []
    for name in impl_names:
        impl_dir, color = IMPLS[name]
        data = [get_time(base_dir, impl_dir, s) for s in sizes]
        if any(d is not None for d in data):
            datasets.append({
                "label": name,
                "data": data,
                "borderColor": color,
                "fill": False,
                "spanGaps": True
            })
    chart_config = {
        "type": "line",
        "data": {"labels": sizes, "datasets": datasets},
        "options": {
            "title": {"display": True, "text": title},
            "scales": {
                "yAxes": [{"scaleLabel": {"display": True, "labelString": "Time (ns)"}}],
                "xAxes": [{"scaleLabel": {"display": True, "labelString": "Size"}}]
            }
        }
    }
    encoded = urllib.parse.quote(json.dumps(chart_config, separators=(',', ':')))
    return f"https://quickchart.io/chart?c={encoded}"

three_impls = ["small-map (FxHash)", "FxHashMap", "micromap"]
all_impls = list(IMPLS.keys())

# Chart 1: 4, 12, 20, 32
print("=== Chart 1: Small scale (4, 12, 20, 32) ===")
print(make_chart_url("Map Performance (4-32)", [4, 12, 20, 32], BASE_64, three_impls))

# Chart 2: 4, 12, 20, 32, 48, 64
print("\n=== Chart 2: Medium scale (4-64) ===")
print(make_chart_url("Map Performance (4-64)", [4, 12, 20, 32, 48, 64], BASE_64, three_impls))

# Chart 3: All maps, 4, 8, 16, 32, 64, 96, 128 from 0~196 Scale
print("\n=== Chart 3: All maps (4-128) ===")
print(make_chart_url("All Maps Performance (4-128)", [4, 8, 16, 32, 64, 96, 128], BASE_196, all_impls))
PYTHON_SCRIPT
