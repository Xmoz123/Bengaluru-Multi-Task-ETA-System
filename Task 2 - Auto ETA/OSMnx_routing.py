"""
OSMnx Road Network and Time-Aware Routing Pipeline
Author: Pratheek Shanbhogue
Task 2: Auto-rickshaw ETA Prediction

Downloads Bengaluru road network and builds time-aware routing graphs
with slot-specific speed adjustments for realistic ETA predictions.
"""

import os
import joblib
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# Configuration
AREA_NAME = "Bengaluru, India"
GRAPH_PKL = "bengaluru_graph_simple.pkl"
GRAPH_CACHE = {}  # In-memory cache for time-aware graphs


def build_base_road_network(area_name: str = AREA_NAME, 
                            save_path: str = GRAPH_PKL) -> nx.MultiDiGraph:
    """
    Download and build base road network for Bengaluru.
    
    Steps:
    1. Download OSM road network (drive network)
    2. Add edge lengths if missing
    3. Keep largest strongly connected component
    4. Save as pickle for fast loading
    
    Returns:
        NetworkX MultiDiGraph with road network
    """
    print(f" Downloading road network for {area_name}...")
    
    # Configure OSMnx
    ox.settings.use_cache = True
    ox.settings.log_console = True
    
    # Download drive network
    G = ox.graph.graph_from_place(
        area_name, 
        network_type='drive', 
        simplify=True
    )
    
    print(f"   Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
    
    # Ensure edge lengths (meters) are present
    has_len = any('length' in data for _, _, data in G.edges(data=True))
    if not has_len:
        G = ox.distance.add_edge_lengths(G)
        print("   Added edge lengths via ox.distance.add_edge_lengths")
    
    # Keep largest strongly connected component
    scc_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(scc_nodes).copy()
    print(f"   After SCC: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Save graph
    joblib.dump(G, save_path)
    print(f"   Saved graph to {save_path}")
    
    return G


def build_time_aware_graph(base_graph_pkl: str,
                          speed_df: pd.DataFrame,
                          date_str: str,
                          slot: int) -> nx.DiGraph:
    """
    Build time-aware routing graph with slot-specific speeds.
    
    Integrates observed speed data for a specific time slot with
    road network to enable realistic routing time predictions.
    
    Args:
        base_graph_pkl: Path to base graph pickle
        speed_df: DataFrame with columns: u, v, slot (speed in km/h)
        date_str: Date string for caching
        slot: Time slot (0-95, 15-min intervals)
        
    Returns:
        DiGraph with travel_time_min attribute on edges
    """
    # Check cache first
    key = (date_str, slot)
    if key in GRAPH_CACHE:
        return GRAPH_CACHE[key]
    
    # Load base graph
    with open(base_graph_pkl, 'rb') as f:
        G = joblib.load(f)
    
    # Build speed lookup from observed data
    speed_map = {}
    cols = list(speed_df.columns)
    
    try:
        u_idx = cols.index('u')
        v_idx = cols.index('v')
    except ValueError:
        raise RuntimeError("speed_df must contain 'u' and 'v' columns")
    
    # Find slot column (could be int or string)
    slot_col = slot if slot in speed_df.columns else str(slot) if str(slot) in speed_df.columns else None
    
    if slot_col is not None:
        s_idx = cols.index(slot_col)
        for row in speed_df.itertuples(index=False, name=None):
            u = row[u_idx]
            v = row[v_idx]
            s = row[s_idx]
            speed_map[(u, v)] = float(s) if pd.notna(s) else np.nan
    
    # Class-based speed priors (km/h) for roads without observations
    priors = {
        'motorway': 55.0,
        'trunk': 45.0,
        'primary': 35.0,
        'secondary': 28.0,
        'tertiary': 24.0,
        'residential': 18.0,
        'service': 14.0
    }
    
    def canonical_highway(hval):
        """Normalize highway tag to single string"""
        if isinstance(hval, (list, tuple)):
            for cand in hval:
                if isinstance(cand, str) and cand in priors:
                    return cand
            return str(hval[0]) if len(hval) else 'residential'
        if isinstance(hval, str):
            return hval
        return 'residential'
    
    # Attach travel times to edges
    for u, v, k, data in G.edges(keys=True, data=True):
        road_class = canonical_highway(data.get('highway', 'residential'))
        length_m = float(data.get('length', 30.0))
        km = max(0.001, length_m / 1000.0)
        
        # Use observed speed or fallback to prior
        sp = speed_map.get((u, v), np.nan)
        if not np.isfinite(sp) or sp <= 0:
            sp = priors.get(road_class, 18.0)
        
        # Calculate travel time (minutes)
        data['travel_time_min'] = km / max(1e-3, sp) * 60.0
    
    # Cache and return
    GRAPH_CACHE[key] = G
    return G


def route_time_minutes(G: nx.DiGraph,
                      orig_xy: tuple,
                      dest_xy: tuple) -> float:
    """
    Calculate routing time between two points using time-aware graph.
    
    Args:
        G: Time-aware graph with travel_time_min edge attribute
        orig_xy: Origin (lat, lon) tuple
        dest_xy: Destination (lat, lon) tuple
        
    Returns:
        Travel time in minutes (0.5-300.0 range)
    """
    try:
        # Find nearest nodes (OSMnx expects x=lon, y=lat)
        on = ox.distance.nearest_nodes(G, orig_xy[1], orig_xy[0])
        dn = ox.distance.nearest_nodes(G, dest_xy[1], dest_xy[0])
        
        # Find shortest path by time
        path = nx.shortest_path(
            G, on, dn, 
            weight='travel_time_min', 
            method='dijkstra'
        )
        
        # Sum travel times along path
        t = 0.0
        for u, v in zip(path[:-1], path[1:]):
            ed = G.get_edge_data(u, v)
            if isinstance(ed, dict):  # Handle parallel edges
                tt = min(d.get('travel_time_min', 0.3) for d in ed.values())
            else:
                tt = ed.get('travel_time_min', 0.3)
            t += float(tt)
        
        return float(np.clip(t, 0.5, 300.0))
    
    except Exception:
        # Robust fallback if routing fails
        return 15.0


# Example usage
if __name__ == "__main__":
    print("OSMnx Road Network & Routing Pipeline")
    print("\n Capabilities:")
    print("   - Download Bengaluru road network")
    print("   - Build time-aware graphs with slot-specific speeds")
    print("   - Route time calculation with realistic travel times")
    print("\n Usage:")
    print("   G_base = build_base_road_network()")
    print("   G_time = build_time_aware_graph(pkl, speed_df, date, slot)")
    print("   time = route_time_minutes(G_time, origin, destination)")
