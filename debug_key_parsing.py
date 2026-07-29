"""Standalone debug script for key parsing logic."""

import numpy as np
from pathlib import Path

# Simulate the key parsing logic
def test_key_parsing():
    # Config values
    horizons = tuple([15, 60, 120])
    thresholds = tuple([10.0])
    
    # Simulated keys from target file
    test_keys = [
        'event_flag_h15_t10.0',
        'future_low_h15_t10.0', 
        'future_high_h15_t10.0',
        'event_start_offset_h15_t10.0',
        'maturity_offset_h15_t10.0',
        'event_flag_h60_t10.0',
        'future_low_h60_t10.0',
        'future_high_h60_t10.0',
        'event_start_offset_h60_t10.0',
        'maturity_offset_h60_t10.0',
        'event_flag_h120_t10.0',
        'future_low_h120_t10.0',
        'future_high_h120_t10.0',
        'event_start_offset_h120_t10.0',
        'maturity_offset_h120_t10.0',
    ]
    
    print("=== Key Parsing Test ===")
    print(f"Config horizons: {horizons} (type: {type(horizons)})")
    print(f"Config thresholds: {thresholds} (type: {type(thresholds)})")
    print()
    
    parsed_results = {}
    for key in test_keys:
        parts = key.split("_")
        if len(parts) >= 3:
            field = parts[0]
            horizon = None
            threshold = None
            for part in parts[1:]:
                if part.startswith("h"):
                    try:
                        horizon = int(part[1:])
                    except ValueError:
                        continue
                elif part.startswith("t"):
                    try:
                        threshold = float(part[1:])
                    except ValueError:
                        continue
            
            if horizon is not None and threshold is not None:
                match = horizon in horizons and threshold in thresholds
                print(f"Key: {key}")
                print(f"  Field: {field}, Horizon: {horizon}, Threshold: {threshold}")
                print(f"  Match: {match}")
                print(f"  horizon in horizons: {horizon in horizons}")
                print(f"  threshold in thresholds: {threshold in thresholds}")
                print()
                
                if match:
                    if field not in parsed_results:
                        parsed_results[field] = []
                    parsed_results[field].append((horizon, threshold))
    
    print("=== Parsed Results ===")
    for field, values in parsed_results.items():
        print(f"{field}: {values}")
    
    return parsed_results

if __name__ == "__main__":
    test_key_parsing()
