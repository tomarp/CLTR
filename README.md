# Chrono‑Light Thermal Response (CL‑TR) Dataset

## Synopsis

The Chrono‑Light Thermal Response (CL‑TR) dataset contains multimodal measurements from a controlled warm‑exposure experiment using a **2 × 2 factorial design** that crossed **light intensity** (Bright vs. Dim) with **time‑of‑day** (Morning vs. Midday). Each participant completed **four sessions**, one in each lighting–time combination, while exposed to moderately warm indoor conditions (air temperature ≈ 30 °C) and a standardized summer clothing ensemble.

Across sessions, we recorded **continuous physiological responses** (e.g., skin temperature, electrodermal activity, heart‑rate–related signals) together with **environmental parameters** and repeated **thermal‑sensation and comfort ratings**. A structured sequence of fan‑cooling and re‑warming phases was used to induce transient and steady‑state thermal loads under otherwise stable room conditions.

The dataset is designed to support research on thermal comfort, thermo‑physiology, light and circadian influences, and dynamic building control strategies.

---

## Experimental Design

### Study Overview

| **Aspect** | **Details** |
|-----------|-------------|
| **Study design** | 2 × 2 within‑subject factorial: **Light Intensity** (Bright vs. Dim) × **Time‑of‑Day** (Morning vs. Midday). Each participant experienced all four conditions in separate sessions. |
| **Participants** | 20 healthy adults (10 M / 10 F), aged 19–30 years. Participants were screened for good general health, normal sleep–wake rhythm, absence of recent jet‑lag or shift work, and acclimatization to the local summer climate. |
| **Sessions per participant** | 4 sessions (≈ 3 h 5 min of structured phases per session), one in each condition: BRI‑MOR, BRI‑MID, DIM‑MOR, DIM‑MID. |
| **Independent variables** | **Light intensity**: Bright vs. Dim, implemented via ceiling luminaires; **Time‑of‑day**: Morning (≈ 07:00–10:30) vs. Midday (≈ 14:00–17:30). |
| **Thermal environment** | Warm but controlled indoor climate (air temperature ≈ 30 °C) in a windowless climate chamber. Clothing was standardized summer attire (≈ 0.3 clo). |
| **Key dependent measures** | **Physiological**: continuous skin temperature, electrodermal activity, and other cardiovascular/thermal proxies. **Subjective**: repeated ratings of thermal sensation, comfort, acceptability, and perceived control. **Behavioural**: fan‑speed changes during participant‑controlled phases. |
| **Design type** | Within‑subject, repeated‑measures design with counterbalanced order of light–time conditions across participants. |
| **Target applications** | Thermal comfort modelling, adaptive and resilience‑based comfort metrics, chrono‑physiology and light, and intelligent building control policies. |

### Light–Time Conditions

Each session is labelled with a compound **condition code** that combines light level and time‑of‑day:

| **Condition code** | **Light level** | **Time‑of‑day window** | **Description** |
|--------------------|-----------------|------------------------|-----------------|
| `BRI‑MOR` | Bright | Morning | Bright light in the morning. |
| `BRI‑MID` | Bright | Midday | Bright light in the early afternoon. |
| `DIM‑MOR` | Dim | Morning | Dim light in the morning. |
| `DIM‑MID` | Dim | Midday | Dim light in the early afternoon. |

The **timeline metadata file (`timeline_metadata.csv`)** accompanying this README lists, for each participant and session, the exact wall‑clock start and end times of all experimental phases, together with their condition codes.

---

## Session Protocol and Phases

Each experimental session followed an identical temporal structure. Participants first underwent a warm **adaptation period**, followed by **three warm exposure blocks**. Within each warm block, a fixed sequence of fan‑cooling and re‑warming phases was implemented in order to induce dynamic changes in skin temperature and thermal sensation under controlled boundary conditions.

At the session level, the total duration of all structured phases was **185 minutes (3 h 5 min)**.

### High‑Level Session Structure

1. **B0 – Adaptation (40 min)**  
   Participants sat quietly in the chamber under the target thermal and lighting condition, allowing physiological variables to stabilize after entry and instrumentation.

2. **Blocks 1–3 – Warm Exposure Blocks**  
   Each block consisted of a repeated four‑phase sequence manipulating fan‑driven convective cooling and subsequent re‑warming:  
   - **Fan‑at‑constant‑speed** (cooling onset)  
   - **Skin‑rewarming** (fan off)  
   - **Fan‑free‑control** (participant can adjust fan)  
   - **Steady‑state period** (fan setting held constant)

   Blocks 1 and 2 used a longer steady‑state period (30 min), while Block 3 used a shorter steady‑state (10 min) to limit overall experiment length.

### Experiment Phase Structure (Relative Timing)

The table below shows the **relative timing** of phases within a session, in minutes from the start of B0. These values are identical for all participants; only the absolute clock times differ between sessions.

| **Phase** | **Duration (min)** | **Start (min)** | **End (min)** | **Notes** |
|----------|--------------------|-----------------|---------------|-----------|
| **B0 – Adaptation** | 40 | 0 | 40 | Initial warm adaptation under fixed light and thermal condition. |
| **Block 1 (total)** | 55 | 40 | 95 | First warm block with long steady‑state. |
| B1.1 – Fan‑at‑constant‑speed | 5 | 40 | 45 | Fan switched to fixed speed (cooling onset). |
| B1.2 – Skin‑rewarming | 10 | 45 | 55 | Fan turned off, passive re‑warming. |
| B1.3 – Fan‑free‑control | 10 | 55 | 65 | Participant can adjust fan speed freely. |
| B1.4 – Steady‑state period | 30 | 65 | 95 | Fan setting fixed; steady‑state warm exposure. |
| **Block 2 (total)** | 55 | 95 | 150 | Second warm block, identical structure to Block 1. |
| B2.1 – Fan‑at‑constant‑speed | 5 | 95 | 100 | Fan switched to fixed speed (cooling onset). |
| B2.2 – Skin‑rewarming | 10 | 100 | 110 | Fan turned off, passive re‑warming. |
| B2.3 – Fan‑free‑control | 10 | 110 | 120 | Participant can adjust fan speed freely. |
| B2.4 – Steady‑state period | 30 | 120 | 150 | Fan setting fixed; steady‑state warm exposure. |
| **Block 3 (total)** | 35 | 150 | 185 | Third warm block with shorter steady‑state. |
| B3.1 – Fan‑at‑constant‑speed | 5 | 150 | 155 | Fan switched to fixed speed (cooling onset). |
| B3.2 – Skin‑rewarming | 10 | 155 | 165 | Fan turned off, passive re‑warming. |
| B3.3 – Fan‑free‑control | 10 | 165 | 175 | Participant can adjust fan speed freely. |
| B3.4 – Steady‑state period | 10 | 175 | 185 | Shortened steady‑state warm exposure. |

These phase codes (e.g., `B1.2 Skin-Rewarm`, `B2.4 Steady-state`) are exactly those used in the accompanying **timeline metadata** and can be used to segment physiological and environmental time series into homogeneous analysis windows.

---

## Timeline Metadata File (`timeline_metadata.csv`)

To facilitate reproducible analysis and cohort definition, the markdown timeline has been converted into a **machine‑readable CSV file** with one row per phase, per participant, per session.

Each row contains:

- `participant_id` – Anonymous participant ID (e.g., `P01`).  
- `participant_label` – Pseudonym (3‑letter code) used in internal documentation.  
- `session_index` – Session number (1–4) for that participant.  
- `light_condition` – Condition code (`BRI‑MOR`, `BRI‑MID`, `DIM‑MOR`, `DIM‑MID`).  
- `session_date` – Calendar date of the session (YYYY‑MM‑DD).  
- `phase_code` – Compact phase identifier (e.g., `B0`, `B1.2`).  
- `block` – Block label (`B0`, `B1`, `B2`, `B3`).  
- `subphase` – Sub‑phase index within a block (1–4 where applicable).  
- `phase_label` – Human‑readable label (e.g., `Fan-Const`, `Skin-Rewarm`).  
- `phase_category` – Canonical, lower‑case category label (e.g., `fan_const`, `skin_rewarm`).  
- `start_time`, `end_time` – Session‑local clock times (HH:MM:SS).  
- `start_iso`, `end_iso` – Combined date–time stamps in ISO‑8601 format (`YYYY‑MM‑DDTHH:MM:SS`).  
- `duration_min` – Phase duration in minutes.  
- `phase_index` – Zero‑based order index of the phase within each participant–session.

This file can serve as a **metadata index** for:
- filtering participants (e.g., by condition or available sessions),  
- extracting specific phases (e.g., all steady‑state segments), and  
- aligning physiological signals, perception ratings, and environmental measurements on a common session/phase timeline.

Researchers can join this metadata with physiological data tables using participant/session identifiers and timestamps to construct consistent analysis cohorts.

---

## Notes for Dataset Users

- All times are expressed in local wall‑clock time and are already aligned within each session; minor differences in absolute start times across participants reflect practical scheduling constraints.  
- Phase durations in the metadata CSV are computed from the provided start and end times and match the nominal structure shown in the phase‑structure table above.  
- When constructing analysis windows, we recommend explicitly selecting both **phase category** (e.g., `steady_state`) and **block** (e.g., `B1`, `B2`, `B3`), as physiological adaptation may differ between early and late warm blocks even under identical nominal conditions.

