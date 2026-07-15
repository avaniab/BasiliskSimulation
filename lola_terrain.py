"""
lola_terrain.py

Small helper for pulling a local lunar surface radius out of the LOLA GDR
polar DEM products (imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/), so the landing
simulation can check touchdown against real terrain instead of a mean sphere.

The polar products are PDS3 images (.IMG + matching .LBL) in a polar
stereographic projection centered on the pole, with pixel values that are
either a signed height/radius already, or a scaled DN that must be converted
with the .LBL's SCALING_FACTOR / OFFSET keys. This reads the label itself
rather than hard-coding those numbers, since they differ between the 4, 10,
20, 40, 60, 118, 128 px/degree products.

Requires:
    pip install pvl planetaryimage pyproj numpy

If those aren't installed, or no DEM path is given, get_surface_radius_m()
falls back to the IAU mean lunar radius (1737400 m) and says so -- it will
never silently pretend to have terrain it doesn't have.
"""

import numpy as np

MEAN_RADIUS_M = 1737400.0  # IAU mean lunar radius, used as fallback

_dem_cache = {}
_fromstring_patched = False


def _patch_numpy_fromstring_binary_mode():
    global _fromstring_patched
    if _fromstring_patched:
        return

    original_fromstring = np.fromstring

    def _compat_fromstring(a, dtype=float, count=-1, sep="", **kwargs):
        if sep == "" and isinstance(a, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(a, dtype=dtype, count=count)
            return arr.copy()
        return original_fromstring(a, dtype=dtype, count=count, sep=sep, **kwargs)

    np.fromstring = _compat_fromstring
    _fromstring_patched = True


def _load(dem_path):
    if dem_path in _dem_cache:
        return _dem_cache[dem_path]

    import pvl
    from planetaryimage import PDS3Image

    _patch_numpy_fromstring_binary_mode()

    img = PDS3Image.open(dem_path)
    label = img.label

    # LOLA polar GDR labels store the map projection under IMAGE_MAP_PROJECTION
    proj = label["IMAGE_MAP_PROJECTION"]
    center_lat = float(proj["CENTER_LATITUDE"].value)
    center_lon = float(proj["CENTER_LONGITUDE"].value)
    map_scale_km = float(proj["MAP_SCALE"].value)  # km/pixel
    line_proj_offset = float(proj["LINE_PROJECTION_OFFSET"].value)
    samp_proj_offset = float(proj["SAMPLE_PROJECTION_OFFSET"].value)
    a_axis_km = float(proj["A_AXIS_RADIUS"].value)

    scaling_factor = float(label.get("SCALING_FACTOR", 1.0))
    offset = float(label.get("OFFSET", 0.0))

    entry = dict(
        data=img.image,
        center_lat=center_lat,
        center_lon=center_lon,
        map_scale_m=map_scale_km * 1000.0,
        line_proj_offset=line_proj_offset,
        samp_proj_offset=samp_proj_offset,
        a_axis_m=a_axis_km * 1000.0,
        scaling_factor=scaling_factor,
        offset=offset,
    )
    _dem_cache[dem_path] = entry
    return entry


def get_surface_radius_m(lat_deg, lon_deg, dem_path=None):
    """
    Returns (radius_m, source_string).

    radius_m is the local lunar radius (center-of-mass to surface) at the
    given planetocentric latitude/longitude, in meters.
    source_string tells you whether that came from real DEM data or the
    fallback mean sphere, so you never mistake one for the other.
    """
    if dem_path is None:
        return MEAN_RADIUS_M, "fallback: IAU mean radius (no DEM supplied)"

    try:
        entry = _load(dem_path)
    except Exception as exc:  # missing deps, bad path, unparseable label, etc.
        return MEAN_RADIUS_M, f"fallback: IAU mean radius (DEM load failed: {exc})"

    import pyproj

    hemisphere = "north" if entry["center_lat"] > 0 else "south"
    src_str = f"+proj=longlat +R={entry['a_axis_m']} +units=m"
    proj_str = (
        f"+proj=stere +lat_0={entry['center_lat']} +lon_0={entry['center_lon']} "
        f"+R={entry['a_axis_m']} +units=m"
    )
    transformer = pyproj.Transformer.from_crs(src_str, proj_str, always_xy=True)
    # LOLA GDR longitudes are planetocentric East, 0-360
    lon_east = lon_deg % 360.0
    x_m, y_m = transformer.transform(lon_east, lat_deg)

    sample = x_m / entry["map_scale_m"] + entry["samp_proj_offset"] + 1
    line = entry["line_proj_offset"] + 1 - y_m / entry["map_scale_m"]

    data = entry["data"]
    if data.ndim == 3:
        data = data[0]
    n_lines, n_samples = data.shape

    if not (0 <= sample < n_samples - 1 and 0 <= line < n_lines - 1):
        return MEAN_RADIUS_M, "fallback: IAU mean radius (lat/lon outside DEM tile)"

    # bilinear interpolation
    s0, l0 = int(np.floor(sample)), int(np.floor(line))
    ds, dl = sample - s0, line - l0
    dn = (
        data[l0, s0] * (1 - ds) * (1 - dl)
        + data[l0, s0 + 1] * ds * (1 - dl)
        + data[l0 + 1, s0] * (1 - ds) * dl
        + data[l0 + 1, s0 + 1] * ds * dl
    )

    height_m = dn * entry["scaling_factor"] + entry["offset"]
    radius_m = MEAN_RADIUS_M + height_m
    return radius_m, f"DEM: {dem_path}"


if __name__ == "__main__":
    # quick smoke test with the fallback path (no DEM installed here)
    r, src = get_surface_radius_m(-89.5, 0.0)
    print(f"radius={r} m  source={src}")


def get_local_slope_deg(lat_deg, lon_deg, dem_path=None, window_m=50.0):
    """
    Estimates local terrain slope (degrees from horizontal) at a lat/lon by
    finite-differencing elevation over a small window (default 50 m).

    Returns (slope_deg, source_string). If no DEM is given or the DEM lookup
    falls back to the mean sphere, slope is reported as 0.0 with a fallback
    note -- that is an OPTIMISTIC assumption (a mean sphere is perfectly
    flat), not a real safety margin. Don't treat a fallback 0 deg as "safe."
    """
    if dem_path is None:
        return 0.0, "fallback: no DEM given, assuming flat (NOT a safety guarantee)"

    R = MEAN_RADIUS_M
    dlat_deg = np.degrees((window_m / 2.0) / R)
    coslat = max(np.cos(np.radians(lat_deg)), 1e-6)
    dlon_deg = np.degrees((window_m / 2.0) / (R * coslat))

    r_n, src_n = get_surface_radius_m(lat_deg + dlat_deg, lon_deg, dem_path)
    r_s, src_s = get_surface_radius_m(lat_deg - dlat_deg, lon_deg, dem_path)
    r_e, src_e = get_surface_radius_m(lat_deg, lon_deg + dlon_deg, dem_path)
    r_w, src_w = get_surface_radius_m(lat_deg, lon_deg - dlon_deg, dem_path)

    if "fallback" in src_n or "fallback" in src_s or "fallback" in src_e or "fallback" in src_w:
        return 0.0, "fallback: window fell outside DEM tile, assuming flat (NOT a safety guarantee)"

    dz_dlat = (r_n - r_s) / window_m
    dz_dlon = (r_e - r_w) / window_m
    slope_rad = np.arctan(np.hypot(dz_dlat, dz_dlon))
    return np.degrees(slope_rad), f"DEM: {dem_path} ({window_m:.0f} m window)"


def assess_landing_stability(slope_deg, horizontal_speed_mps, vertical_speed_mps,
                              leg_radius_m, cg_height_m, attitude_tilt_deg,
                              g_local): 
    theta_crit_deg = np.degrees(np.arctan(leg_radius_m / cg_height_m)) 
    v_char = np.sqrt(max(2.0 * g_local * cg_height_m, 1e-6)) #characteristic velocity 
    dynamic_tilt_deg = np.degrees(np.arctan(horizontal_speed_mps / v_char)) 

    effective_tilt_deg = slope_deg + abs(attitude_tilt_deg) + dynamic_tilt_deg 
    margin_deg = theta_crit_deg - effective_tilt_deg 

    return {
        "theta_crit_deg": theta_crit_deg,
        "slope_deg": slope_deg,
        "dynamic_tilt_deg": dynamic_tilt_deg,
        "effective_tilt_deg": effective_tilt_deg,
        "margin_deg": margin_deg,
        "stable": margin_deg > 0.0, #TODO threshold??? 
    }

    #NOTE -5.4609 kg/min rate 