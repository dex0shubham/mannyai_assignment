import os
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Order:
    """
    Encapsulates a production order with timing and constraints.
    """
    def __init__(self, row):
        self.id = row['order_id']
        self.num_items = int(row.get('num_items', 1))
        self.product_type = row['product_type']
        self.cut_time = row['cut_time'] * self.num_items
        self.sew_time = row['sew_time'] * self.num_items
        self.pack_time = row['pack_time'] * self.num_items
        self.deadline = row['deadline']
        self.post_cut_delay = 48 if row.get('post_cut_delay', False) else 0


class Machine:
    """
    Models a single machine in the workflow.
    """
    def __init__(self, stage, idx):
        self.id = f"{stage}-{idx}"
        self.available_at = 0
        self.last_product = None
        self.utilization = []  # List of (start, end, order_id)

    def schedule(self, earliest, duration, order_id, setup=0, product_type=None):
        """
        Schedule a job on this machine, applying setup time.
        Returns actual start and end.
        """
        start = max(self.available_at + setup, earliest)
        end = start + duration
        self.available_at = end
        if product_type is not None:
            self.last_product = product_type
        self.utilization.append((start, end, order_id))
        return start, end


def load_orders(path: str) -> pd.DataFrame:
    """
    Loads orders from Excel (.xls/.xlsx) or CSV (*.csv).
    Normalizes columns and ensures defaults.
    """
    ext = path.lower()
    if ext.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path, engine='openpyxl')
    elif ext.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input file type: {path}")

    df = df.rename(columns={
        'Product type': 'product_type',
        'cut time': 'cut_time',
        'sew time': 'sew_time',
        'pack time': 'pack_time',
        'requires_out_of_factory_delay': 'post_cut_delay'
    })
    if 'num_items' not in df.columns:
        df['num_items'] = 1
    df['post_cut_delay'] = df['post_cut_delay'].astype(bool)
    return df


def simulate_schedule(orders, n_cut, n_sew, n_pack):
    """
    Forward schedules orders through Cut→Sew→Pack.
    Returns schedule dict, avg lateness, on-time count, machine lists.
    """
    cutting = [Machine('Cut', i+1) for i in range(n_cut)]
    sewing  = [Machine('Sew', i+1) for i in range(n_sew)]
    packing = [Machine('Pack', i+1) for i in range(n_pack)]

    schedule = {}
    total_late = 0
    on_time = 0

    for o in orders:
        # Cut
        opts = []
        for m in cutting:
            setup = 10 if (m.last_product and m.last_product != o.product_type) else 0
            ready = m.available_at + setup
            opts.append((ready, m, setup))
        t_cut, m_cut, setup = min(opts, key=lambda x: x[0])
        cut_s, cut_e = m_cut.schedule(t_cut, o.cut_time, o.id, setup, o.product_type)

        # Sew
        t_ready_sew = cut_e + o.post_cut_delay
        opts = [(max(m.available_at, t_ready_sew), m) for m in sewing]
        t_sew, m_sew = min(opts, key=lambda x: x[0])
        sew_s, sew_e = m_sew.schedule(t_sew, o.sew_time, o.id)

        # Pack
        opts = [(max(m.available_at, sew_e), m) for m in packing]
        t_pack, m_pack = min(opts, key=lambda x: x[0])
        pack_s, pack_e = m_pack.schedule(t_pack, o.pack_time, o.id)

        lateness = max(0, pack_e - o.deadline)
        total_late += lateness
        on_time += (lateness == 0)

        schedule[o.id] = {'Cut': (m_cut.id, cut_s, cut_e),
                          'Sew': (m_sew.id, sew_s, sew_e),
                          'Pack': (m_pack.id, pack_s, pack_e),
                          'Lateness': lateness}

    avg_lateness = total_late / len(orders)
    return schedule, avg_lateness, on_time, cutting, sewing, packing


def generate_permutation(orders):
    """
    Random permutation based on deadline/product/processing heuristics.
    """
    r = random.random()
    if r < 0.6:
        key, name = (lambda o: o.deadline), 'deadline'
    elif r < 0.8:
        key, name = (lambda o: o.product_type), 'product'
    else:
        key, name = (lambda o: o.cut_time+o.sew_time+o.pack_time), 'processing'

    groups = {}
    for o in sorted(orders, key=key):
        groups.setdefault(key(o), []).append(o)
    perm = []
    for grp in groups.values():
        random.shuffle(grp)
        perm.extend(grp)
    return perm, name


def optimize(orders, iterations, machines, seed, weight):
    """
    Monte Carlo optimizing a weighted cost:
      cost = weight*avg_lateness - (1-weight)*on_time
    Lower cost is better.
    Returns best dict and history.
    """
    random.seed(seed); np.random.seed(seed)
    best = {'cost': float('inf'), 'avg_lateness': None, 'on_time': None, 'schedule': None}
    hist_on, hist_lat, hist_cost, hist_heur = [], [], [], []

    for _ in tqdm(range(iterations), desc='Optimizing'):
        perm, heur = generate_permutation(orders[:])
        sched, avg_late, on_time, *_ = simulate_schedule(perm, *machines)
        cost = weight*avg_late - (1-weight)*on_time
        hist_on.append(on_time); hist_lat.append(avg_late)
        hist_cost.append(cost); hist_heur.append(heur)
        if cost < best['cost']:
            best.update(cost=cost, avg_lateness=avg_late, on_time=on_time, schedule=sched)

    return best, hist_on, hist_lat, hist_cost, hist_heur


def plot_gantt(schedule, path=None):
    rows = {}
    for oid, stg in schedule.items():
        for phase in ['Cut','Sew','Pack']:
            m, s, e = stg[phase]
            rows.setdefault(m, []).append((s, e, oid))
    fig, ax = plt.subplots(figsize=(12,6))
    for idx, (m, tasks) in enumerate(sorted(rows.items())):
        for s, e, oid in tasks:
            ax.barh(idx, e-s, left=s, edgecolor='black', alpha=0.6)
            ax.text(s+(e-s)/2, idx, str(oid), va='center', ha='center', fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(sorted(rows.keys()))
    ax.set_xlabel('Time Units'); ax.set_title('Machine Gantt Chart')
    plt.tight_layout()
    if path: plt.savefig(path)
    else: plt.show()


def plot_lateness_dist(schedule, path=None):
    ids = list(schedule.keys()); lates = [schedule[i]['Lateness'] for i in ids]
    x = range(len(ids))
    fig, ax = plt.subplots()
    ax.bar(x, lates)
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=90)
    ax.set_xlabel('Order ID'); ax.set_ylabel('Lateness')
    ax.set_title('Lateness Distribution')
    plt.tight_layout()
    if path: plt.savefig(path)
    else: plt.show()


def plot_progress(hist_on, hist_lat, hist_cost, path=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(hist_on, label='On-Time', linewidth=1)
    ax2.plot(hist_lat, '--', label='Avg Lateness', linewidth=1)
    ax2.plot(hist_cost, ':', label='Cost', linewidth=1)
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('On-Time Orders')
    ax2.set_ylabel('Avg Lateness / Cost'); ax1.set_title('Optimization Progress')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout()
    if path: plt.savefig(path)
    else: plt.show()


def plot_heuristic_dist(hist_heur, path=None):
    counts = {h: hist_heur.count(h) for h in set(hist_heur)}
    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
    ax.set_title('Heuristic Usage')
    if path: plt.savefig(path)
    else: plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Excel or CSV file of orders')
    p.add_argument('--cut', type=int, default=2, help='Cutting tables')
    p.add_argument('--sew', type=int, default=3, help='Sewing machines')
    p.add_argument('--pack', type=int, default=1, help='Packing stations')
    p.add_argument('-n', type=int, default=500, help='Monte Carlo iterations')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--weight', type=float, default=0.5,
                   help='Weight for avg lateness in cost function (0..1)')
    p.add_argument('--out', default='results', help='Output folder')
    args = p.parse_args()

    df = load_orders(args.input)
    orders = [Order(r) for _, r in df.iterrows()]
    machines = (args.cut, args.sew, args.pack)

    best, hist_on, hist_lat, hist_cost, hist_heur = optimize(
        orders, args.n, machines, args.seed, args.weight
    )

    # ——— build a lateness summary DataFrame ———
    sched = best['schedule']
    summary_df = pd.DataFrame([
        {'order_id': oid, 'lateness': info['Lateness']}
        for oid, info in sched.items()
    ])
    summary_df['on_time'] = summary_df['lateness'] == 0

    # ——— print per‐order lateness ———
    print("\nSummary of lateness by order:")
    print(summary_df.set_index('order_id'))

    # ——— compute and print the two aggregates ———
    total_on_time = summary_df['on_time'].sum()
    n_orders     = len(summary_df)
    # assuming time-units are hours; divide by 24 for days
    avg_days_late = summary_df.loc[~summary_df['on_time'], 'lateness'].mean() / 24 if total_on_time < n_orders else 0

    print(f"\nTotal orders on time: {total_on_time} / {n_orders}")
    print(f"Average days late (only late ones): {avg_days_late:.2f} days")

    os.makedirs(args.out, exist_ok=True)
    print(f"Weight: {args.weight}")
    print(f"Best cost: {best['cost']:.2f}")
    print(f"Average lateness: {best['avg_lateness']:.2f}")
    print(f"On-time orders: {best['on_time']}/{len(orders)}")

    plot_gantt(best['schedule'], os.path.join(args.out, 'gantt.png'))
    plot_lateness_dist(best['schedule'], os.path.join(args.out, 'lateness_dist.png'))
    plot_progress(hist_on, hist_lat, hist_cost, os.path.join(args.out, 'progress.png'))
    plot_heuristic_dist(hist_heur, os.path.join(args.out, 'heuristic_dist.png'))

    print(f"Charts written to '{args.out}/'")

if __name__ == '__main__':
    main()
