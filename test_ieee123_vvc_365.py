import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gridages.envs.single_agent.scenarios.ieee123_vvc import (
    IEEE123VVCEnv,
)

#Configuration

N_DAYS = 366
HOURS_PER_DAY = 24
N_HOURS = N_DAYS * HOURS_PER_DAY

VMIN_LIMIT = 0.95
VMAX_LIMIT = 1.05
LINE_LIMIT = 100.0

#Build IEEE-123 VVC environment

env = IEEE123VVCEnv(
    env_config = {
        "train": False,
        "episode_length": N_HOURS,
        "reward_scale": 1.0,
        "safety_scale": 0.0,
        "store_info": [],
        "store_summaries": False,
    }
)

#Dataset check

for key in ["load", "solar", "wind", "price"]:

    data = np.asarray(
        env.dataset[key],
        dtype=float,
    )

    print(
        f"{key}: {len(data)} points"
    )

    if len(data) < N_HOURS:
        raise ValueError(
            f"{key} contains only "
            f"{len(data)} points. "
            f"Need {N_HOURS}."
        )
    
    #Keep exactly 2024 data and add ONE dummy point.
    #
    #GridBaseEnv.step() increments t and calls
    #_apply_dataset_scalers() even after the final terminated step.
    #This additional sample prevents an index error.
    #It is not included in the 2024 analysis.

    env.dataset[key] = np.concatenate(
        [
            data[:N_HOURS],
            data[:1],
        ]
    )

#Reset
obs, info = env.reset(seed=0)

print()
print("Initial timestep:", env.t)

if env.t != 0:
    raise RuntimeError(
        f"Expected reset to start at t=0,"
        f"but env.t={env.t}"
    )
    
    #Neutral / no-VVC-control action
    #
    # QPV = 0
    # QWT = 0
    #regulator tap positions = 0
    #capacitor banks = OFF

#Check dataset length

baseline_action = {
    "continuous": np.zeros(
        10,
        dtype=np.float32,
    ),

    "discrete": np.array(
        [
            3, #Regulator 1 -> tap +1``
            16, #Regulator 2 -> tap 0
            16, #Regulator 3 -> tap 0
            2, #Regulator 4 -> tap 0
            1, #SCB 1 -> step 1
            2, #SCB 2 -> off
        ],
        dtype = np.int64,
    ),
}

#Run 8784 hourly operating points using env.step()
records = []

for hour in range(N_HOURS):

    #Current operating point Before env.step()

    current_t = int(env.t)

    load_profile = float(
        env.dataset["load"][current_t]
    )

    solar_profile = float(
        env.dataset["solar"][current_t]
    )

    wind_profile = float(
        env.dataset["wind"][current_t]
    )

    #Actual GridAges environment step
    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    )=env.step(
        baseline_action
    )

    #Solve AC power flow

    converged = bool(
        env.net.converged
    )

    #Extract results

    if converged:

        vm = env.net.res_bus[
            "vm_pu"
        ].to_numpy()

        loading = env.net.res_line[
            "loading_percent"
        ].to_numpy()

        vmin = float(vm.min())
        vmax = float(vm.max())

        undervoltage_buses = int(
            (
                vm < VMIN_LIMIT
            ).sum()
        )

        overvoltage_buses = int(
            (
                vm > VMAX_LIMIT
            ).sum()
        )

        voltage_violation_buses = int(
            undervoltage_buses
            +
            overvoltage_buses
        )

        overloaded_lines = int(
            (loading > LINE_LIMIT).sum()
        )

        max_line_loading = float(
            loading.max()
        )

        line_loss_mw = float(
            env.net.res_line[
                "pl_mw"
            ].sum()
        )

    else:

        vmin = np.nan
        vmax = np.nan
        
        undervoltage_buses = np.nan
        overvoltage_buses = np.nan
        voltage_violation_buses = np.nan
        overloaded_lines = np.nan
        max_line_loading = np.nan
        line_loss_mw = np.nan

        #Store hourly result

    records.append(
        {
            "hour": hour,

            "day": hour // 24 + 1,

            "hour_of_day":
                hour % 24,

            "converged":
                converged,

            "vmin":
                vmin,

            "vmax":
                vmax,

            "undervoltage_buses":
                undervoltage_buses,
            
            "overvoltage_buses":
                overvoltage_buses,

            "voltage_violation_buses":
                voltage_violation_buses,

            "max_line_loading_percent":
                max_line_loading,

            "overloaded_lines":
                overloaded_lines,

            "line_loss_mw":
                line_loss_mw,

            "reward":
                float(reward),

            "load_profile":
                load_profile,

            "solar_profile":
                solar_profile,

            "wind_profile":
                wind_profile,
            
        }
    )

    #Progress

    if (hour+1) % 720 == 0:
        print(
            f"Completed"
            f"{hour + 1}/ {N_HOURS} hours"
        )


#Hourly results

df = pd.DataFrame(
    records
)

print(
     "Recorded hours:",
     len(df)
)

# Feasibility indicators

df["voltage_feasible"] = (
    (df["vmin"] >= VMIN_LIMIT)
    &
    (df["vmax"] <= VMAX_LIMIT)
)

df["voltage_violation"]=(
    df["converged"]
    &
    (
        (df["vmin"] < VMIN_LIMIT)
        |
        (df["vmax"] > VMAX_LIMIT)
    )
)

df["thermal_feasible"] = (
    df[
        "max_line_loading_percent"
    ]
    <= LINE_LIMIT
)

df["fully_feasible"] = (
    df["converged"]
    &
    df["voltage_feasible"]
    &
    df["thermal_feasible"]
)

df.to_csv(
    "ieee123_vvc_2024_hourly_no_control.csv",
    index=False,
)

#Daily summary

daily = (
    df.groupby("day")
    .agg(
        vmin=(
            "vmin",
            "min",
        ),

        vmax = (
            "vmax",
            "max",
        ),

        max_line_loading_percent = (
            "max_line_loading_percent",
            "max",
        ),

        voltage_violation_hours=(
            "voltage_violation",
            "sum",
        ),

        overload_hours=(
            "thermal_feasible",
            lambda x: int(
                (~x).sum()
            ),
        ),

        converged_hours=(
            "converged",
            "sum",
        ),

        feasible_hours=(
            "fully_feasible",
            "sum",
        ),

        mean_line_loss_mw = (
            "line_loss_mw",
            "mean",
        ),
    )
    .reset_index()
)

daily.to_csv(
    "ieee123_vvc_2024_fixed_baseline.csv",
    index=False,
)

print(
    "Recorded hours:",
    len(df)
)

assert len(df) == N_HOURS, (
    f"Expected {N_HOURS} records, "
    f"got {len(df)}."
)

#Overall Benchmark statistics
 
converged_hours = int(
    df["converged"].sum()
)

nonconverged_hours = int(
    (~df["converged"]).sum()
)

voltage_violation_hours = int(
    (
        df[
            "voltage_violation_buses"
        ]
        >0
    ).sum()
)

undervoltage_hours = int(
    (
        df[
            "undervoltage_buses"
        ]
        > 0
    ).sum()
)

overvoltage_hours = int(
    (
        df[
            "overvoltage_buses"
        ]
        > 0
    ).sum()
)

overload_hours = int(
    (
        df[
            "overloaded_lines"
        ]
        > 0
    ).sum()
)
fully_feasible_hours = int(
    df[
        "fully_feasible"
    ].sum()
)

days_with_voltage_violations = int(
    (
        daily[
            "voltage_violation_hours"
        ]
        > 0
    ).sum()
)

worst_vmin = float(
    df["vmin"].min()
)

worst_vmax = float(
    df["vmax"].max()
)

worst_loading = float(
    df[
        "max_line_loading_percent"
    ].max()
)

mean_loss = float(
    df[
        "line_loss_mw"
    ].mean()
)

print()
print(
    "======================================="
)
print(
    "IEEE-123 VVC 2024 NO-CONTROL TEST"
)

print(
    f"Total hours:               "
    f"{N_HOURS}"
)

print(
    f"Converged hours:            "
    f"{converged_hours}"
)

print(
    f"Non-converged hours:        "
    f"{nonconverged_hours}"
)

print(
    f"Fully feasible hours:       "
    f"{fully_feasible_hours}"
)

print(
    f"Voltage violation hours:    "
    f"{voltage_violation_hours}"
)

print(
    f"Undervoltage hours:             "
    f"{undervoltage_hours}"
)

print(
    f"Overvoltage hours:              "
    f"{overvoltage_hours}"
)

print(
    f"Days with voltage violations:   "
    f"{days_with_voltage_violations}"
)

print(
    f"Line overload hours:        "
    f"{overload_hours}"
)

print(
    f"Worst maximum voltage:      "
    f"{worst_vmax:.4f} pu"
)

print(
    f"Worst minimum voltage:      "
    f"{worst_vmin:.4f} pu"
)

print(
    f"Maximum line loading:       "
    f"{worst_loading:.2f} %"
)

print(
    f"Mean line loss:             "
    f"{mean_loss:.6f} MW"
)

#Voltage-versus-day plot
plt.figure(figsize = (12, 5))

plt.plot(
    df["hour"] / 24,
    df["load_profile"],
    linewidth = 0.8,
)
plt.xlabel("Day")
plt.ylabel("Load scaling factor")
plt.title(
    "IEEE-123 Load Profile - 2024"
)
plt.xlim(
    0,
    366,
)
plt.grid(
    True,
    alpha=0.3,
)
plt.tight_layout()

plt.savefig(
    "ieee123_load_profile_2024.png",
    dpi=300,
)
plt.show()

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    daily["day"],
    daily["vmin"],
    label="Daily minimum voltage",
)

plt.plot(
    daily["day"],
    daily["vmax"],
    label="Daily maximum voltage",
)

plt.axhline(
    VMIN_LIMIT,
    linestyle="--",
    label="0.95 pu limit",
)

plt.axhline(
    VMAX_LIMIT,
    linestyle="--",
    label="1.05 pu limit",
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Voltage magnitude (pu)"
)

plt.title(
    "IEEE-123 Fixed-Baseline Voltage Profile - 2024"
)

plt.legend()

plt.grid(
    True,
    alpha = 0.3,
)

plt.tight_layout()

plt.savefig(
    "ieee123_vvc_voltage_vs_day.png",
    dpi=300,
)

plt.show()