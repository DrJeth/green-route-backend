from flask import Flask, render_template, request, jsonify
import osmnx as ox
import networkx as nx
import traceback
import math
from shapely.geometry import Point, LineString
import geopandas as gpd

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Config (tweak if needed)
FORBIDDEN_ROADS = {"primary", "primary_link", "trunk", "trunk_link", "motorway", "motorway_link"}
MAX_ALLOW_CROSSING_M = 6        # recommended: allow tiny crossing only (meters)
ORIGIN_CONNECT_FALLBACK_RADIUS_M = 120  # how far to connect origin to nearest node
ORIGIN_BOUNDARY_TOL_M = 120     # allow origin to be slightly outside boundary due to GPS noise

# --- Load Biñan graph ONCE at startup
print("Loading Biñan graph (this may take a while the first time)...")
G_ORIG = ox.graph_from_place("Biñan, Laguna, Philippines", network_type="drive")
print("Graph loaded. Nodes:", len(G_ORIG.nodes), "Edges:", len(G_ORIG.edges))

def haversine_m(a_lat, a_lon, b_lat, b_lon):
    R = 6371000
    phi1 = math.radians(a_lat)
    phi2 = math.radians(b_lat)
    dphi = math.radians(b_lat - a_lat)
    dlambda = math.radians(b_lon - a_lon)
    aa = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(aa))

def remove_forbidden_long_edges(Gin, max_allow_crossing_m=MAX_ALLOW_CROSSING_M):
    G = Gin.copy()
    for u, v, k, data in list(G.edges(keys=True, data=True)):
        road_type = data.get("highway")
        if isinstance(road_type, list):
            road_type = road_type[0] if road_type else None
        if road_type in FORBIDDEN_ROADS:
            length = data.get("length", float("inf"))
            if length is None:
                length = float("inf")
            if length > max_allow_crossing_m:
                try:
                    G.remove_edge(u, v, k)
                except Exception:
                    pass
    return G

def build_coords_from_path(Gf, path, origin_coord=None, dest_coord=None):
    coords = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = Gf.get_edge_data(u, v) or Gf.get_edge_data(v, u)
        used = False
        if edge_data:
            key = next(iter(edge_data))
            data = edge_data[key]
            geom = data.get("geometry")
            if geom is not None:
                for lon, lat in list(geom.coords):
                    coords.append([lat, lon])
                used = True
        if not used:
            ny = Gf.nodes[u].get("y")
            nx_ = Gf.nodes[u].get("x")
            if ny is not None and nx_ is not None:
                coords.append([ny, nx_])
    # append final node coords
    last = path[-1]
    coords.append([Gf.nodes[last]["y"], Gf.nodes[last]["x"]])

    # dedupe consecutive duplicates
    filtered = []
    prev = None
    for c in coords:
        if prev is None or (abs(prev[0] - c[0]) > 1e-9 or abs(prev[1] - c[1]) > 1e-9):
            filtered.append(c)
        prev = c

    # ensure start/end exactly match requested origin/dest if provided
    if origin_coord:
        if filtered:
            if abs(filtered[0][0] - origin_coord[0]) > 1e-5 or abs(filtered[0][1] - origin_coord[1]) > 1e-5:
                filtered.insert(0, [origin_coord[0], origin_coord[1]])
        else:
            filtered.insert(0, [origin_coord[0], origin_coord[1]])
    if dest_coord:
        if filtered:
            if abs(filtered[-1][0] - dest_coord[0]) > 1e-5 or abs(filtered[-1][1] - dest_coord[1]) > 1e-5:
                filtered.append([dest_coord[0], dest_coord[1]])
        else:
            filtered.append([dest_coord[0], dest_coord[1]])

    return filtered

# --- Load Biñan polygon (server-side) ---
print("Loading Biñan polygon...")
try:
    gdf = ox.geocode_to_gdf("Biñan, Laguna, Philippines")
    if not gdf.empty:
        BOUNDARY_POLY = gdf.unary_union
        print("Biñan polygon loaded.")
    else:
        BOUNDARY_POLY = None
        print("Warning: Biñan polygon not found.")
except Exception as e:
    print("Warning loading boundary polygon:", e)
    BOUNDARY_POLY = None

# --- Clip original graph to Biñan polygon (remove nodes outside) ---
try:
    if BOUNDARY_POLY is not None:
        print("Clipping graph to Biñan polygon (removing nodes outside)...")
        nodes_within = []
        for n, attr in G_ORIG.nodes(data=True):
            x = attr.get("x")
            y = attr.get("y")
            if x is None or y is None:
                continue
            p = Point(x, y)
            if p.within(BOUNDARY_POLY):
                nodes_within.append(n)
        G_CLIPPED = G_ORIG.subgraph(nodes_within).copy()
        print("Clipped graph nodes:", len(G_CLIPPED.nodes), "edges:", len(G_CLIPPED.edges))
    else:
        G_CLIPPED = G_ORIG.copy()
        print("Boundary not available — using full graph without clipping.")
except Exception as e:
    print("Graph clipping error:", e)
    G_CLIPPED = G_ORIG.copy()

# --- Precompute filtered graph (forbidden edges removed) ---
print("Applying forbidden-edge filter to clipped graph...")
G_FILTERED = remove_forbidden_long_edges(G_CLIPPED, max_allow_crossing_m=MAX_ALLOW_CROSSING_M)
print("Filtered graph ready. Nodes:", len(G_FILTERED.nodes), "Edges:", len(G_FILTERED.edges))

@app.route("/route", methods=["GET"])
def get_route():
    try:
        origin_lat = float(request.args.get("origin_lat"))
        origin_lon = float(request.args.get("origin_lon"))
        dest_lat = float(request.args.get("dest_lat"))
        dest_lon = float(request.args.get("dest_lon"))
    except Exception as e:
        return jsonify({"status":"error","message":"bad parameters","detail": str(e)}), 400

    # Server-side check: destination must be inside Biñan
    try:
        if BOUNDARY_POLY is not None:
            pd = Point(dest_lon, dest_lat)
            if not pd.within(BOUNDARY_POLY):
                return jsonify({"status":"outside_boundary","message":"Destination is outside Biñan boundary."}), 200
            po = Point(origin_lon, origin_lat)
            if not po.within(BOUNDARY_POLY):
                # allow origin to be slightly outside due to GPS error (tolerance)
                d_origin = haversine_m(origin_lat, origin_lon, po.y, po.x) if False else None
                # Instead compute nearest node inside polygon and distance:
                try:
                    near_node = ox.distance.nearest_nodes(G_FILTERED, origin_lon, origin_lat)
                    near_attr = G_FILTERED.nodes[near_node]
                    d = haversine_m(origin_lat, origin_lon, near_attr.get("y"), near_attr.get("x"))
                    if d > ORIGIN_BOUNDARY_TOL_M:
                        return jsonify({"status":"outside_boundary_origin","message":"Origin is outside Biñan boundary (too far)."}), 200
                    else:
                        # accept origin but we'll connect via virtual node as usual
                        pass
                except Exception:
                    return jsonify({"status":"outside_boundary_origin","message":"Origin appears outside Biñan boundary."}), 200
    except Exception as e:
        print("Boundary check error:", e)

    try:
        Gf = G_FILTERED.copy()
        dest_node = ox.distance.nearest_nodes(Gf, dest_lon, dest_lat)

        Gtmp = Gf.copy()
        virt_id = "__virt_origin__"
        success_project = False

        try:
            u, v, key = ox.distance.nearest_edges(Gf, origin_lon, origin_lat)
            edge_data = Gf.get_edge_data(u, v) or Gf.get_edge_data(v, u)
            if edge_data:
                k0 = next(iter(edge_data))
                data0 = edge_data[k0]
                geom = data0.get("geometry")
                if geom is None:
                    geom = LineString([(Gf.nodes[u]['x'], Gf.nodes[u]['y']), (Gf.nodes[v]['x'], Gf.nodes[v]['y'])])

                p = Point(origin_lon, origin_lat)
                proj_dist = geom.project(p)
                proj_point = geom.interpolate(proj_dist)
                proj_lon, proj_lat = proj_point.x, proj_point.y
                d_to_proj = haversine_m(origin_lat, origin_lon, proj_lat, proj_lon)

                if d_to_proj < 2000:
                    Gtmp.add_node(virt_id, x=proj_lon, y=proj_lat)

                    try:
                        Gtmp.remove_edge(u, v, k0)
                    except Exception:
                        try:
                            Gtmp.remove_edge(u, v)
                        except Exception:
                            pass

                    coords = list(geom.coords)
                    cum = 0.0
                    split_done = False
                    u_coords = None
                    v_coords = None
                    for i in range(len(coords)-1):
                        seg = LineString([coords[i], coords[i+1]])
                        seg_len = seg.length
                        if cum + seg_len >= proj_dist - 1e-12:
                            u_coords = coords[:i+1] + [(proj_lon, proj_lat)]
                            v_coords = [(proj_lon, proj_lat)] + coords[i+1:]
                            split_done = True
                            break
                        cum += seg_len
                    if not split_done:
                        u_coords = [coords[0], (proj_lon, proj_lat)]
                        v_coords = [(proj_lon, proj_lat), coords[-1]]

                    geom_u_proj = LineString(u_coords)
                    geom_proj_v = LineString(v_coords)

                    def total_meters_from_coords(c_list):
                        total = 0.0
                        for a,b in zip(c_list[:-1], c_list[1:]):
                            total += haversine_m(a[1], a[0], b[1], b[0])
                        return total

                    len_u_proj = total_meters_from_coords(geom_u_proj.coords)
                    len_proj_v = total_meters_from_coords(geom_proj_v.coords)

                    attr = dict(data0)
                    attr.pop("geometry", None)
                    attr.pop("length", None)

                    Gtmp.add_edge(u, virt_id, key=0, geometry=geom_u_proj, length=len_u_proj, **attr)
                    Gtmp.add_edge(virt_id, u, key=0, geometry=LineString(list(geom_u_proj.coords)[::-1]), length=len_u_proj, **attr)
                    Gtmp.add_edge(virt_id, v, key=0, geometry=geom_proj_v, length=len_proj_v, **attr)
                    Gtmp.add_edge(v, virt_id, key=0, geometry=LineString(list(geom_proj_v.coords)[::-1]), length=len_proj_v, **attr)

                    success_project = True
        except Exception as e:
            success_project = False
            print("Projection attempt error (will fallback):", e)

        if not success_project:
            Gtmp.add_node(virt_id, x=origin_lon, y=origin_lat)
            candidates = []
            for n, attr in Gf.nodes(data=True):
                ny = attr.get("y")
                nx_ = attr.get("x")
                if ny is None or nx_ is None:
                    continue
                d = haversine_m(origin_lat, origin_lon, ny, nx_)
                if d <= ORIGIN_CONNECT_FALLBACK_RADIUS_M:
                    candidates.append((n, d))
            if not candidates:
                near = ox.distance.nearest_nodes(Gf, origin_lon, origin_lat)
                d = haversine_m(origin_lat, origin_lon, Gf.nodes[near]["y"], Gf.nodes[near]["x"])
                candidates = [(near, d)]
            candidates.sort(key=lambda x: x[1])
            n, d = candidates[0]
            Gtmp.add_edge(virt_id, n, key=0, length=float(d))
            Gtmp.add_edge(n, virt_id, key=0, length=float(d))

        try:
            path = nx.shortest_path(Gtmp, virt_id, dest_node, weight="length")
        except nx.NetworkXNoPath:
            return jsonify({"status":"no_path","message":"No path found within e-bike allowed network"}), 200

        # SAFETY CHECK: ensure all nodes in path are inside Biñan polygon
        if BOUNDARY_POLY is not None:
            for node in path:
                na = Gtmp.nodes[node]
                nxcoord = na.get("x")
                nycoord = na.get("y")
                if nxcoord is None or nycoord is None:
                    continue
                if not Point(nxcoord, nycoord).within(BOUNDARY_POLY):
                    # found node outside Biñan — reject path
                    return jsonify({"status":"no_path","message":"Computed path exits Biñan boundary — rejected for safety."}), 200

        route_coords = build_coords_from_path(Gtmp, path, origin_coord=[origin_lat, origin_lon], dest_coord=[dest_lat, dest_lon])
        return jsonify({"status":"ok","route": route_coords}), 200

    except Exception as e:
        tb = traceback.format_exc()
        print("Route error:", tb)
        return jsonify({"status":"error","message":"exception building route","detail": str(e)}), 200

if __name__ == "__main__":
    app.run(debug=True)


