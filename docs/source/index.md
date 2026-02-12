# GridAges Documentation

**GridAges** is an agent-centric power grid simulator for **reinforcement learning (RL)** and **multi-agent RL (MARL)**, built on **pandapower**.

- *Agent-centric*: compose microgrids, DER clusters, and distribution grids as modular agents
- *Physics-based*: AC power flow via pandapower
- *RL-friendly*: Gymnasium-compatible single-agent and PettingZoo-compatible multi-agent environments
- *Dataset-driven uncertainty*: load, renewables, and price from real-world time series


---

## Getting Started

```{toctree}
:maxdepth: 2

getting-started/installation
getting-started/dataset
getting-started/ieee13-quickstart

```

## Concepts

```{toctree}
:maxdepth: 2

concepts/overview
concepts/agent-centric
concepts/ems-opf-mdp
concepts/constraints-and-safety
concepts/uncertainty-and-data

```

## API Reference

```{toctree}
:maxdepth: 2

api/envs
api/devices
api/networks

```

## Contributing
```{toctree}
:maxdepth: 2

contributing/contributing
contributing/roadmap

```