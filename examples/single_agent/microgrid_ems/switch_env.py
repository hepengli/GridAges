#!/usr/bin/env python
"""
Helper script to switch between IEEE13 and IEEE34 environments for training.
Usage: python switch_env.py [ieee13|ieee34]
"""

import os
import sys

SCRIPTS = ['train.py', 'train_sac.py', 'train_ddpg.py', 'train_td3.py']

IEEE13_IMPORT = "from gridages.envs.single_agent.microgrid_ems.ieee13_mg import IEEE13Env"
IEEE13_ENV = "env = IEEE13Env(env_config=env_config)"
IEEE13_LOGS = [
    'LOG_DIR = "./logs/ppo"',
    'LOG_DIR = "./logs/sac"',
    'LOG_DIR = "./logs/ddpg"',
    'LOG_DIR = "./logs/td3"',
]
IEEE13_SAVES = [
    'SAVE_DIR = "./models/ppo"',
    'SAVE_DIR = "./models/sac"',
    'SAVE_DIR = "./models/ddpg"',
    'SAVE_DIR = "./models/td3"',
]

IEEE34_IMPORT = "from gridages.envs.single_agent.microgrid_ems.ieee34_mg import IEEE34Env"
IEEE34_ENV = "env = IEEE34Env(env_config=env_config)"
IEEE34_LOGS = [
    'LOG_DIR = "./logs/ppo_ieee34"',
    'LOG_DIR = "./logs/sac_ieee34"',
    'LOG_DIR = "./logs/ddpg_ieee34"',
    'LOG_DIR = "./logs/td3_ieee34"',
]
IEEE34_SAVES = [
    'SAVE_DIR = "./models/ppo_ieee34"',
    'SAVE_DIR = "./models/sac_ieee34"',
    'SAVE_DIR = "./models/ddpg_ieee34"',
    'SAVE_DIR = "./models/td3_ieee34"',
]


def switch_to_ieee13():
    """Switch all training scripts to IEEE13 environment."""
    print("Switching to IEEE13 environment...")
    
    for script, log, save in zip(SCRIPTS, IEEE13_LOGS, IEEE13_SAVES):
        filepath = script
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace imports
        content = content.replace(IEEE34_IMPORT, IEEE13_IMPORT)
        content = content.replace(IEEE34_ENV, IEEE13_ENV)
        
        # Replace log directories
        for ieee34_log in IEEE34_LOGS:
            if ieee34_log.split('"')[1] in content:
                idx = IEEE34_LOGS.index(ieee34_log)
                content = content.replace(ieee34_log, IEEE13_LOGS[idx])
        
        # Replace save directories
        for ieee34_save in IEEE34_SAVES:
            if ieee34_save.split('"')[1] in content:
                idx = IEEE34_SAVES.index(ieee34_save)
                content = content.replace(ieee34_save, IEEE13_SAVES[idx])
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"  ✓ {script}")
    
    print("\n✓ Switched to IEEE13!")


def switch_to_ieee34():
    """Switch all training scripts to IEEE34 environment."""
    print("Switching to IEEE34 environment...")
    
    for script, log, save in zip(SCRIPTS, IEEE34_LOGS, IEEE34_SAVES):
        filepath = script
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace imports
        content = content.replace(IEEE13_IMPORT, IEEE34_IMPORT)
        content = content.replace(IEEE13_ENV, IEEE34_ENV)
        
        # Replace log directories
        for ieee13_log in IEEE13_LOGS:
            if ieee13_log.split('"')[1] in content:
                idx = IEEE13_LOGS.index(ieee13_log)
                content = content.replace(ieee13_log, IEEE34_LOGS[idx])
        
        # Replace save directories
        for ieee13_save in IEEE13_SAVES:
            if ieee13_save.split('"')[1] in content:
                idx = IEEE13_SAVES.index(ieee13_save)
                content = content.replace(ieee13_save, IEEE34_SAVES[idx])
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"  ✓ {script}")
    
    print("\n✓ Switched to IEEE34!")


def show_current():
    """Show which environment is currently configured."""
    with open(SCRIPTS[0], 'r') as f:
        content = f.read()
    
    if 'IEEE34Env' in content:
        print("Current environment: IEEE34")
    else:
        print("Current environment: IEEE13")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python switch_env.py [ieee13|ieee34|status]")
        print("\nOptions:")
        print("  ieee13  - Switch to IEEE13 environment")
        print("  ieee34  - Switch to IEEE34 environment")
        print("  status  - Show current environment")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'ieee13':
        switch_to_ieee13()
    elif cmd == 'ieee34':
        switch_to_ieee34()
    elif cmd == 'status':
        show_current()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
