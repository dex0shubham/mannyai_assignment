# Schedule Optimizer

A Monte Carlo–based production scheduler that assigns orders to cutting, sewing, and packing machines to minimize average lateness and maximize on‑time completions. It also produces per‑order lateness summaries, overall on‑time counts, and average lateness.

---

## Features

- Reads orders from Excel or CSV, with configurable item counts and factory delays.
- Forward schedules through Cut → Sew → Pack stages with machine setup times.
- Monte Carlo optimization using deadline, product, or processing time heuristics.
- Gantt chart of machine utilization.
- Lateness distribution bar chart.
- Optimization progress (on‑time, average lateness, cost).
- Heuristic usage pie chart.
- Console summary of per‑order lateness, total on‑time orders, and average lateness.

---

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

- pandas >= 1.0
- numpy >= 1.18
- matplotlib >= 3.0
- tqdm >= 4.0
- openpyxl >= 3.0

---

## Usage

```bash
python more_schedule.py C:\Users\rhira\MannyAI\pythonProject\data.xlsx --cut 2 --sew 3 --pack 1 -n 5000 --out C:\Users\rhira\MannyAI\pythonProject\results
```

- `<input_file>`: Path to Excel (.xls/.xlsx) or CSV file of orders.

Optional arguments:

| Flag        | Description                                              | Default              |
|-------------|----------------------------------------------------------|----------------------|
| `--cut N`   | Number of cutting tables                                 | 2                    |
| `--sew N`   | Number of sewing machines                                | 3                    |
| `--pack N`  | Number of packing stations                               | 1                    |
| `-n N`      | Monte Carlo iterations                                   | 500 (override: 5000) |
| `--seed N`  | Random seed                                              | 42                   |
| `--weight w`| Weight for average lateness in cost function (0..1)      | 0.5                  |
| `--out DIR` | Output folder for charts and results summary             | results              |

-------------|----------------------------------------------------------|---------|
| `--cut N`   | Number of cutting tables                                 | 2       |
| `--sew N`   | Number of sewing machines                                | 3       |
| `--pack N`  | Number of packing stations                               | 1       |
| `-n N`      | Monte Carlo iterations                                   | 500     |
| `--seed N`  | Random seed                                              | 42      |
| `--weight w`| Weight for average lateness in cost function (0..1)      | 0.5     |
| `--out DIR` | Output folder for charts and results summary             | results |

---

## Example Output

After running, you will see something like:

```
Summary of lateness by order:
          lateness  on_time
order_id
O016             0     True
O012             0     True
O031             0     True
O037            15    False
O032             0     True
O002             0     True
O040            66    False
O043             0     True
O014             0     True
O044             0     True
O020             0     True
O041             0     True
O011            62    False
O042             0     True
O023             0     True
O008             0     True
O005             0     True
O006           103    False
O026             0     True
O039            31    False
O046           147    False
O013            27    False
O035            57    False
O021             0     True
O009             0     True
O024           111    False
O033           123    False
O027           193    False
O010           121    False
O003           202    False
O047           257    False
O025           131    False
O049             0     True
O018            89    False
O045           139    False
O050             0     True
O028            48    False
O036           141    False
O019           326    False
O029           173    False
O007           372    False
O038           182    False
O022             0     True
O048           124    False
O030           274    False
O015           378    False
O001           134    False
O017           417    False
O004           295    False
O034           431    False

Total orders on time: 20 / 50

Average lateness: 103.38
On-time orders: 20/50
```

Charts are written to the output directory (`--out`).

---

## Interpretation

- **Per-order lateness** shows how late each order finished relative to its deadline.
- **Total orders on time** is the count of orders with zero lateness.
- **Average lateness** is computed over all orders (in the same time units as your input).

---

## Approach and Steps

1. **Data Loading**: Read the input orders from an Excel or CSV file, normalize column names, and apply defaults (e.g., `num_items`, `post_cut_delay`).
2. **Order & Machine Modeling**: 
   - **Order** objects encapsulate processing times (cut, sew, pack), deadlines, and any post-cut delay.
   - **Machine** objects track availability, last product type (for setup times), and utilization schedule.
3. **Schedule Simulation**:
   - For a given sequence of orders, iteratively assign each to the earliest available cutting, sewing, and packing machine.
   - Apply setup times if switching product types on a machine.
   - Record start/end times per stage and compute lateness (`max(0, finish_time - deadline)`).
4. **Monte Carlo Optimization**:
   - **Heuristics**: generate many random permutations of the order list using three methods: deadlines first, product grouping, or total processing time.
   - **Simulation**: for each permutation, simulate the schedule and collect:
     1. **Average Lateness**: the mean lateness across all orders.
     2. **On-Time Count**: the number of orders that finished by their deadline.
   - **Cost Function**: combine these metrics into a single cost value:
     ```text
     cost = weight * avg_lateness - (1 - weight) * on_time_count
     ```
     where `weight` ∈ [0,1] balances minimizing lateness vs. maximizing on-time orders.
   - **Selection**: keep the permutation with the lowest cost as the **best** schedule.
5. **Result Aggregation**:
   - Build a per-order lateness summary DataFrame.
   - Calculate total on-time orders and average lateness (optionally converted to days).
6. **Visualization & Reporting**:
   - Output console summary: per-order lateness, total on-time count, average lateness.
   - Generate and save charts: Gantt chart, lateness distribution, optimization progress, heuristic usage.

---

## Code Flow

```text
main()
 ├─ load_orders()
 │   └─ read Excel/CSV into DataFrame
 ├─ wrap rows in Order objects
 ├─ optimize()
 │   ├─ for each iteration:
 │   │   ├─ generate_permutation()
 │   │   ├─ simulate_schedule()
 │   │   │   ├─ loop orders:
 │   │   │   │   ├─ schedule cut on Machine
 │   │   │   │   ├─ schedule sew on Machine
 │   │   │   │   ├─ schedule pack on Machine
 │   │   │   │   └─ compute lateness
 │   │   │   └─ return schedule & stats
 │   │   └─ compute cost & update best
 │   └─ return best schedule & history
 ├─ aggregate_results()
 │   ├─ build lateness DataFrame
 │   ├─ compute on-time count
 │   └─ compute avg days late
 ├─ print summaries
 └─ plot_*() functions for charts
```

---

## Output Files

- `gantt.png`: Machine Gantt chart.
- `lateness_dist.png`: Bar chart of lateness per order.
- `progress.png`: Optimization progress over iterations.
- `heuristic_dist.png`: Pie chart of heuristic usage.

---

