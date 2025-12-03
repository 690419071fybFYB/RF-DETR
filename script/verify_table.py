
from rfdetr.main import print_metrics_table

stats = {
    "results_json": {
        "class_map": [
            {
                "class": "plane",
                "map@50:95": 0.85,
                "map@50": 0.95,
                "precision": 0.90,
                "recall": 0.88
            },
            {
                "class": "car",
                "map@50:95": 0.75,
                "map@50": 0.85,
                "precision": 0.80,
                "recall": 0.78
            }
        ]
    }
}

print("Testing print_metrics_table...")
print_metrics_table(stats, title="Test Metrics")
