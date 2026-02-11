#!/usr/bin/env python
"""Test script to verify IEEE34Env can be instantiated with the data file."""

import sys
sys.path.insert(0, '/Users/yanbinlin/GridAges')

from gridages.envs.single_agent.microgrid_ems.ieee34_mg import IEEE34Env

print("Testing IEEE34Env initialization...")

env_config = {
    "load_scale": 1.0,
    "reward_scale": 1.0,
    "safety_scale": 10000.0,
    "max_penalty": 10000.0,
    "train": True,
    "store_info": ["operating_cost", "bus_voltage", "line_loading"],
    "store_summaries": True,
    "store_arrays": False,
}

try:
    env = IEEE34Env(env_config=env_config)
    print("✓ IEEE34Env initialized successfully!")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Try a reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful!")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Info keys: {list(info.keys())}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
