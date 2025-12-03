
from rfdetr import RFDETRBase
from rfdetr.config import RFDETRBaseConfig

# Test case 1: Default behavior (use_dynamic_query=True)
print("Testing default behavior (use_dynamic_query=True)...")
rf_default = RFDETRBase()
if hasattr(rf_default.model.model, 'query_importance_head'):
    print("PASS: query_importance_head exists by default.")
else:
    print("FAIL: query_importance_head missing by default.")

# Test case 2: Disable dynamic query (use_dynamic_query=False)
print("\nTesting disabled dynamic query (use_dynamic_query=False)...")
config = RFDETRBaseConfig(use_dynamic_query=False)
rf_static = RFDETRBase(use_dynamic_query=False)

if not hasattr(rf_static.model.model, 'query_importance_head'):
    print("PASS: query_importance_head does not exist when disabled.")
else:
    print("FAIL: query_importance_head exists when disabled.")

if not hasattr(rf_static.model.model, 'query_budget_predictor'):
    print("PASS: query_budget_predictor does not exist when disabled.")
else:
    print("FAIL: query_budget_predictor exists when disabled.")
