import pandas as pd
import numpy as np

train_df = pd.read_csv('data/processed/splits/train.csv')
train_df['date'] = pd.to_datetime(train_df['date'])
pivot_df = train_df.pivot(index='date', columns='reservoir', values='inflow_day_1')
corr_matrix = pivot_df.corr(method='pearson', min_periods=365) # minimum 1 year overlap

overlap_counts = pivot_df.notna().astype(int).T.dot(pivot_df.notna().astype(int))
min_overlap = overlap_counts.values.min()

def analyze_graph(corr_matrix, threshold):
    reservoirs = list(corr_matrix.columns)
    edges = []
    adj = {r: [] for r in reservoirs}
    
    for i in range(len(reservoirs)):
        for j in range(i+1, len(reservoirs)):
            r1 = reservoirs[i]
            r2 = reservoirs[j]
            c = corr_matrix.loc[r1, r2]
            if not np.isnan(c) and c >= threshold:
                edges.append((r1, r2, c))
                adj[r1].append(r2)
                adj[r2].append(r1)
                
    degrees = [len(adj[r]) for r in reservoirs]
    isolated = [r for r in reservoirs if len(adj[r]) == 0]
    
    # Components
    visited = set()
    components = []
    for r in reservoirs:
        if r not in visited:
            stack = [r]
            comp = set()
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.add(node)
                    stack.extend(adj[node])
            components.append(sorted(list(comp)))
            
    n = len(reservoirs)
    max_possible_edges = n * (n - 1) / 2
    density = len(edges) / max_possible_edges
    
    print(f"\n--- Threshold {threshold} ---")
    print(f"Edges: {len(edges)}")
    print(f"Connected components: {len(components)}")
    print(f"Isolated nodes: {len(isolated)} ({isolated})")
    print(f"Min degree: {min(degrees)}")
    print(f"Max degree: {max(degrees)}")
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Density: {density:.3f}")

print(f"Min overlap across all pairs: {min_overlap}")
for t in [0.55]:
    analyze_graph(corr_matrix, t)

