from dotenv import load_dotenv
import os

load_dotenv()

import math
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ==========================================================
# EcoRoute Demo App
# Real-time cost-efficient routing with carbon impact insights
# ==========================================================

st.set_page_config(page_title="EcoRoute", page_icon="🌱", layout="wide")

# -----------------------------
# Judge-ready UI styling
# -----------------------------
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f766e 0%, #16a34a 55%, #84cc16 100%);
        padding: 2rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.22);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        line-height: 1.05;
    }
    .hero-tagline {
        font-size: 1.35rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        max-width: 950px;
    }
    .value-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.1rem;
        min-height: 125px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    .value-card h4 {
        margin: 0 0 0.45rem 0;
        color: #0f172a;
    }
    .value-card p {
        margin: 0;
        color: #334155;
        font-size: 0.95rem;
    }
    .section-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 1.6rem 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.9rem;
        color: #475569;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
    }
    .judge-note {
        background: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        color: #064e3b;
        margin-top: 1rem;
    }
    .small-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #334155;
        margin-top: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sample Suffolk-area demo data
# Coordinates are stored as (latitude, longitude)
# -----------------------------
LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Depot - Ipswich": (52.0567, 1.1482),
    "Customer A - Kesgrave": (52.0616, 1.2347),
    "Customer B - Woodbridge": (52.0932, 1.3189),
    "Customer C - Martlesham": (52.0622, 1.2836),
    "Customer D - Felixstowe": (51.9630, 1.3511),
    "Customer E - Stowmarket": (52.1894, 0.9977),
}

# -----------------------------
# Cost and emissions assumptions
# -----------------------------
VEHICLE_PROFILES = {
    "Petrol Van": {
        "base_emission_kg_per_km": 0.21,
        "base_cost_per_km": 0.22,
        "idle_cost_per_hour": 2.60,
    },
    "Diesel Van": {
        "base_emission_kg_per_km": 0.24,
        "base_cost_per_km": 0.20,
        "idle_cost_per_hour": 2.40,
    },
    "Hybrid Van": {
        "base_emission_kg_per_km": 0.15,
        "base_cost_per_km": 0.17,
        "idle_cost_per_hour": 1.70,
    },
    "Electric Van": {
        "base_emission_kg_per_km": 0.02,
        "base_cost_per_km": 0.12,
        "idle_cost_per_hour": 0.80,
    },
}

# Payload affects fuel/energy demand. Mapbox handles traffic-aware routing,
# so traffic is not requested manually in the UI.
BASELINE_PAYLOAD_KG = 100
MAX_PAYLOAD_KG = 1000
PAYLOAD_IMPACT_PER_100KG = 0.04


def payload_multiplier(payload_kg: float) -> float:
    """
    Estimate the impact of payload weight on fuel/energy use.

    Baseline = 100 kg. For every additional 100 kg above baseline,
    fuel/energy demand, cost, and emissions increase by approximately 4%.
    Below baseline, impact is capped at 1.00 for simplicity.
    """
    extra_payload = max(0.0, payload_kg - BASELINE_PAYLOAD_KG)
    return 1 + ((extra_payload / 100) * PAYLOAD_IMPACT_PER_100KG)


ROUTE_EFFICIENCY = {
    "Standard": 1.08,
    "EcoRoute": 0.96,
}


# -----------------------------
# Core helper functions
# -----------------------------
def latlon_to_lonlat(point: Tuple[float, float]) -> str:
    """Convert (lat, lon) to Mapbox string format 'lon,lat'."""
    lat, lon = point
    return f"{lon},{lat}"


def haversine_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Fallback straight-line distance between two latitude/longitude points."""
    lat1, lon1 = p1
    lat2, lon2 = p2

    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def fallback_route_distance_km(route_names: List[str]) -> float:
    """Fallback route distance when Mapbox is unavailable."""
    total = 0.0
    for i in range(len(route_names) - 1):
        total += haversine_km(LOCATIONS[route_names[i]], LOCATIONS[route_names[i + 1]])
    return total


def build_standard_route(stops: List[str]) -> List[str]:
    """Standard route follows the user-selected stop order and returns to depot."""
    return ["Depot - Ipswich"] + stops + ["Depot - Ipswich"]


def build_fallback_eco_route(stops: List[str]) -> List[str]:
    """Nearest-neighbour fallback when Mapbox token/API is unavailable."""
    current = "Depot - Ipswich"
    unvisited = stops.copy()
    route = [current]

    while unvisited:
        next_stop = min(
            unvisited,
            key=lambda x: haversine_km(LOCATIONS[current], LOCATIONS[x]),
        )
        route.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    route.append("Depot - Ipswich")
    return route


def calculate_route_impact(
    distance_km: float,
    duration_minutes: float,
    vehicle_type: str,
    payload_kg: float,
    route_type: str,
) -> dict:
    """
    Cost-first route impact model.

    Mapbox provides real road distance and traffic-aware duration.
    EcoRoute then converts that route into business impact using vehicle type,
    payload weight, route duration, and route efficiency.
    """
    vehicle = VEHICLE_PROFILES[vehicle_type]
    duration_hours = duration_minutes / 60

    load_multiplier = payload_multiplier(payload_kg)
    route_efficiency = ROUTE_EFFICIENCY[route_type]

    running_cost = distance_km * vehicle["base_cost_per_km"] * load_multiplier * route_efficiency

    # Traffic impact is already reflected in Mapbox duration, so no manual traffic input is needed.
    congestion_cost = duration_hours * vehicle["idle_cost_per_hour"]
    total_cost = running_cost + congestion_cost

    co2_kg = (
        distance_km
        * vehicle["base_emission_kg_per_km"]
        * load_multiplier
        * route_efficiency
    )

    return {
        "road_distance_km": round(distance_km, 2),
        "duration_minutes": round(duration_minutes, 1),
        "payload_kg": round(payload_kg, 1),
        "payload_multiplier": round(load_multiplier, 3),
        "co2_kg": round(co2_kg, 2),
        "cost_gbp": round(total_cost, 2),
    }


def annualised_saving(value_per_route: float, operating_days: int) -> float:
    return value_per_route * operating_days


# -----------------------------
# Mapbox integration
# -----------------------------
def get_mapbox_token(user_token: str) -> Optional[str]:
    """Read token from sidebar input, Streamlit secrets, or environment variable."""
    if user_token:
        return user_token.strip()

    try:
        token = st.secrets.get("MAPBOX_ACCESS_TOKEN")
        if token:
            return token
    except Exception:
        pass

    return os.getenv("MAPBOX_ACCESS_TOKEN")


@st.cache_data(ttl=300, show_spinner=False)
def get_mapbox_directions(
    route_names: List[str],
    token: str,
    profile: str = "mapbox/driving-traffic",
) -> dict:
    """
    Get a traffic-aware route for a fixed waypoint order using Mapbox Directions API.
    Returns distance, duration, and GeoJSON coordinates.
    """
    coordinates = ";".join([latlon_to_lonlat(LOCATIONS[name]) for name in route_names])
    url = f"https://api.mapbox.com/directions/v5/{profile}/{coordinates}"
    params = {
        "access_token": token,
        "geometries": "geojson",
        "overview": "full",
        "steps": "false",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(data.get("message", "Mapbox Directions API did not return a valid route."))

    route = data["routes"][0]
    return {
        "distance_km": route["distance"] / 1000,
        "duration_minutes": route["duration"] / 60,
        "geometry": route["geometry"],
        "source": "Mapbox Directions API",
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_mapbox_optimized_route(
    stops: List[str],
    token: str,
    profile: str = "mapbox/driving-traffic",
) -> dict:
    """
    Get a traffic-aware optimized multi-stop route using Mapbox Optimization API.
    The first coordinate is the depot and the trip is treated as a round trip.
    """
    input_names = ["Depot - Ipswich"] + stops
    coordinates = ";".join([latlon_to_lonlat(LOCATIONS[name]) for name in input_names])
    url = f"https://api.mapbox.com/optimized-trips/v1/{profile}/{coordinates}"
    params = {
        "access_token": token,
        "geometries": "geojson",
        "overview": "full",
        "roundtrip": "true",
        "source": "first",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if data.get("code") != "Ok" or not data.get("trips"):
        raise ValueError(data.get("message", "Mapbox Optimization API did not return a valid route."))

    trip = data["trips"][0]

    ordered_waypoints = sorted(data["waypoints"], key=lambda x: x.get("waypoint_index", 0))
    route_order = []
    for wp in ordered_waypoints:
        original_index = data["waypoints"].index(wp)
        if original_index < len(input_names):
            route_order.append(input_names[original_index])

    if not route_order or route_order[0] != "Depot - Ipswich":
        route_order = ["Depot - Ipswich"] + [name for name in route_order if name != "Depot - Ipswich"]
    if route_order[-1] != "Depot - Ipswich":
        route_order.append("Depot - Ipswich")

    return {
        "route_order": route_order,
        "distance_km": trip["distance"] / 1000,
        "duration_minutes": trip["duration"] / 60,
        "geometry": trip["geometry"],
        "source": "Mapbox Optimization API",
    }


def fallback_route_result(route_names: List[str], route_type: str) -> dict:
    """Fallback route result without live Mapbox data."""
    distance = fallback_route_distance_km(route_names)
    estimated_speed_kmh = 42 if route_type == "Standard" else 45
    return {
        "route_order": route_names,
        "distance_km": distance,
        "duration_minutes": (distance / estimated_speed_kmh) * 60,
        "geometry": None,
        "source": "Fallback heuristic",
    }


# -----------------------------
# Lightweight ML emissions model
# -----------------------------
@st.cache_resource
def train_emissions_model() -> tuple:
    training_rows = []

    for vehicle_type in VEHICLE_PROFILES.keys():
        for payload_kg in range(50, MAX_PAYLOAD_KG + 1, 50):
            for route_type in ROUTE_EFFICIENCY.keys():
                for distance_km in range(5, 151, 5):
                    duration_minutes = (distance_km / 40) * 60
                    impact = calculate_route_impact(
                        distance_km=float(distance_km),
                        duration_minutes=duration_minutes,
                        vehicle_type=vehicle_type,
                        payload_kg=float(payload_kg),
                        route_type=route_type,
                    )
                    training_rows.append(
                        {
                            "road_distance_km": float(distance_km),
                            "duration_minutes": duration_minutes,
                            "vehicle_type": vehicle_type,
                            "payload_kg": float(payload_kg),
                            "route_type": route_type,
                            "co2_kg": impact["co2_kg"],
                        }
                    )

    df = pd.DataFrame(training_rows)
    features = [
        "road_distance_km",
        "duration_minutes",
        "vehicle_type",
        "payload_kg",
        "route_type",
    ]
    target = "co2_kg"

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.25, random_state=42
    )

    categorical_features = ["vehicle_type", "route_type"]
    numeric_features = ["road_distance_km", "duration_minutes", "payload_kg"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=150, random_state=42, max_depth=8)),
        ]
    )

    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    return model, score, len(df)


def predict_ml_co2(
    model: Pipeline,
    road_distance_km: float,
    duration_minutes: float,
    vehicle_type: str,
    payload_kg: float,
    route_type: str,
) -> float:
    input_df = pd.DataFrame(
        [
            {
                "road_distance_km": road_distance_km,
                "duration_minutes": duration_minutes,
                "vehicle_type": vehicle_type,
                "payload_kg": payload_kg,
                "route_type": route_type,
            }
        ]
    )
    return float(model.predict(input_df)[0])


def explain_route_factors(vehicle_type: str, payload_kg: float, route_type: str) -> List[str]:
    explanations = []

    if vehicle_type in ["Diesel Van", "Petrol Van"]:
        explanations.append("Fuel vehicle profile increases route emissions and operating cost.")
    elif vehicle_type == "Electric Van":
        explanations.append("Electric vehicle profile reduces direct emissions, but route efficiency still affects energy cost.")
    else:
        explanations.append("Hybrid vehicle profile reduces emissions compared with petrol/diesel.")

    if payload_kg > BASELINE_PAYLOAD_KG:
        explanations.append(
            f"A {payload_kg:.0f} kg payload increases energy demand compared with the {BASELINE_PAYLOAD_KG} kg baseline."
        )
    else:
        explanations.append(
            f"A {payload_kg:.0f} kg payload is close to or below the baseline, keeping energy demand lower."
        )

    if route_type == "EcoRoute":
        explanations.append("EcoRoute uses Mapbox route optimisation and lower route-efficiency penalties.")
    else:
        explanations.append("Standard route follows the manual stop order and may be less efficient.")

    return explanations


# -----------------------------
# Map drawing
# -----------------------------
def draw_route_map(
    standard_route: List[str],
    eco_route: List[str],
    standard_geometry: Optional[dict],
    eco_geometry: Optional[dict],
) -> None:
    if not FOLIUM_AVAILABLE:
        st.info("Map visualisation requires: pip install folium streamlit-folium")
        return

    m = folium.Map(location=[52.06, 1.20], zoom_start=10)

    def geojson_to_latlon(geometry: dict) -> List[Tuple[float, float]]:
        return [(lat, lon) for lon, lat in geometry["coordinates"]]

    if standard_geometry:
        standard_coords = geojson_to_latlon(standard_geometry)
    else:
        standard_coords = [LOCATIONS[name] for name in standard_route]

    if eco_geometry:
        eco_coords = geojson_to_latlon(eco_geometry)
    else:
        eco_coords = [LOCATIONS[name] for name in eco_route]

    folium.PolyLine(
        standard_coords,
        tooltip="Standard route",
        popup="Standard route",
        color="red",
        weight=5,
        opacity=0.75,
    ).add_to(m)

    folium.PolyLine(
        eco_coords,
        tooltip="EcoRoute",
        popup="EcoRoute",
        color="green",
        weight=5,
        opacity=0.9,
        dash_array="8, 8",
    ).add_to(m)

    all_route_places = set(standard_route + eco_route)
    for name in all_route_places:
        icon_colour = "blue" if name == "Depot - Ipswich" else "gray"
        folium.Marker(
            location=LOCATIONS[name],
            popup=name,
            tooltip=name,
            icon=folium.Icon(color=icon_colour, icon="info-sign"),
        ).add_to(m)

    st_folium(m, width=1050, height=560)


def route_table_from_order(route_names: List[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(route_names) - 1):
        rows.append({"Stop": i + 1, "From": route_names[i], "To": route_names[i + 1]})
    return pd.DataFrame(rows)


# -----------------------------
# Session state
# -----------------------------
if "analysis_has_run" not in st.session_state:
    st.session_state.analysis_has_run = False


# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🌱 EcoRoute</div>
        <div class="hero-tagline">Real-time cost-efficient routing with carbon impact insights</div>
        <div class="hero-subtitle">
            A last-mile delivery optimisation platform that helps SME fleets reduce operating costs first, while making carbon savings visible, measurable, and reportable.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="value-card">
            <h4>💰 Cost-first</h4>
            <p>Prioritises delivery cost reduction so the business case is immediate and measurable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="value-card">
            <h4>🛰️ Real-time routing</h4>
            <p>Uses Mapbox road-network and traffic-aware routing when a valid API token is supplied.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="value-card">
            <h4>📦 Payload-aware impact</h4>
            <p>Uses delivery load weight in kg to estimate fuel, cost and carbon impact more realistically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.caption(
    "Tagline: Real-time cost-efficient routing with carbon impact insights. "
    "Mapbox provides live road-network routing using the driving-traffic profile. "
    "EcoRoute adds fleet-specific cost, payload and carbon impact logic on top."
)

ml_model, ml_r2_score, ml_training_rows = train_emissions_model()

with st.expander("ℹ️ What this app now solves", expanded=False):
    st.markdown(
        """
EcoRoute now has three layers:

1. **Real road-network routing layer** — uses Mapbox Directions/Optimization APIs when a token is available.
2. **Cost-first optimisation layer** — prioritises operating cost, then shows CO₂ reduction.
3. **ML emissions prediction layer** — predicts route emissions from distance, duration, vehicle type, payload weight, and route type.

Manual traffic input has been removed because Mapbox already provides traffic-aware route distance and duration. Payload remains as a user input because Mapbox does not know how heavy the delivery vehicle is.
        """
    )

with st.expander("🤖 ML component", expanded=False):
    st.markdown(
        f"""
The prototype includes a lightweight ML emissions prediction layer trained on **{ml_training_rows} synthetic route scenarios**.

Indicative validation score:

```text
R² score: {ml_r2_score:.3f}
```

In production, this model would be retrained using real fleet data: GPS routes, fuel/energy usage, vehicle telematics, payload weight, delivery times, and driver behaviour.
        """
    )

with st.sidebar:
    st.header("Demo Controls")

    token_input = st.text_input(
        "Mapbox access token",
        type="password",
        help="Optional. Add a Mapbox public access token to enable live traffic-aware routing.",
    )
    mapbox_token = get_mapbox_token(token_input)

    vehicle = st.selectbox("Vehicle type", list(VEHICLE_PROFILES.keys()), index=0)

    payload_kg = st.number_input(
        "Delivery payload weight (kg)",
        min_value=0,
        max_value=MAX_PAYLOAD_KG,
        value=250,
        step=25,
        help="Estimated total delivery load carried by the vehicle. Mapbox handles traffic; payload helps EcoRoute estimate cost and CO₂ impact.",
    )

    selected_stops = st.multiselect(
        "Select delivery stops",
        options=[k for k in LOCATIONS.keys() if k != "Depot - Ipswich"],
        default=[
            "Customer A - Kesgrave",
            "Customer B - Woodbridge",
            "Customer C - Martlesham",
            "Customer D - Felixstowe",
        ],
    )

    operating_days = st.slider("Operating days per year", 100, 365, 260)

    if st.button("Run EcoRoute Analysis", type="primary"):
        st.session_state.analysis_has_run = True

    if st.button("Reset Analysis"):
        st.session_state.analysis_has_run = False

run_demo = st.session_state.analysis_has_run

if not selected_stops:
    st.warning("Please select at least one delivery stop to run the demo.")
    st.stop()

if run_demo:
    standard_route = build_standard_route(selected_stops)
    api_status = "Fallback mode"
    mapbox_warning = None

    # -----------------------------
    # Route generation
    # -----------------------------
    if mapbox_token:
        try:
            standard_result = get_mapbox_directions(standard_route, mapbox_token)
            eco_result = get_mapbox_optimized_route(selected_stops, mapbox_token)
            eco_route = eco_result["route_order"]
            api_status = "Live Mapbox mode: traffic-aware routing enabled"
        except Exception as exc:
            mapbox_warning = str(exc)
            eco_route = build_fallback_eco_route(selected_stops)
            standard_result = fallback_route_result(standard_route, "Standard")
            eco_result = fallback_route_result(eco_route, "EcoRoute")
    else:
        eco_route = build_fallback_eco_route(selected_stops)
        standard_result = fallback_route_result(standard_route, "Standard")
        eco_result = fallback_route_result(eco_route, "EcoRoute")

    # -----------------------------
    # Impact calculation
    # -----------------------------
    standard_impact = calculate_route_impact(
        distance_km=standard_result["distance_km"],
        duration_minutes=standard_result["duration_minutes"],
        vehicle_type=vehicle,
        payload_kg=payload_kg,
        route_type="Standard",
    )

    eco_impact = calculate_route_impact(
        distance_km=eco_result["distance_km"],
        duration_minutes=eco_result["duration_minutes"],
        vehicle_type=vehicle,
        payload_kg=payload_kg,
        route_type="EcoRoute",
    )

    cost_saved = max(0.0, standard_impact["cost_gbp"] - eco_impact["cost_gbp"])
    co2_saved = max(0.0, standard_impact["co2_kg"] - eco_impact["co2_kg"])
    co2_reduction_pct = (co2_saved / standard_impact["co2_kg"] * 100) if standard_impact["co2_kg"] > 0 else 0

    annual_cost_saved = annualised_saving(cost_saved, operating_days)
    annual_co2_saved = annualised_saving(co2_saved, operating_days)

    standard_ml_co2 = predict_ml_co2(
        ml_model,
        standard_impact["road_distance_km"],
        standard_impact["duration_minutes"],
        vehicle,
        payload_kg,
        "Standard",
    )
    eco_ml_co2 = predict_ml_co2(
        ml_model,
        eco_impact["road_distance_km"],
        eco_impact["duration_minutes"],
        vehicle,
        payload_kg,
        "EcoRoute",
    )
    ml_co2_saved = max(0.0, standard_ml_co2 - eco_ml_co2)
    ml_co2_reduction_pct = (ml_co2_saved / standard_ml_co2 * 100) if standard_ml_co2 > 0 else 0

    # -----------------------------
    # Results
    # -----------------------------
    st.markdown("## 💼 Impact Summary: Cost First, Carbon Visible")
    st.caption(api_status)
    if mapbox_warning:
        st.warning(f"Mapbox unavailable, using fallback heuristic: {mapbox_warning}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Standard CO₂", f"{standard_impact['co2_kg']:.2f} kg")
    col2.metric("EcoRoute CO₂", f"{eco_impact['co2_kg']:.2f} kg", delta=f"-{co2_reduction_pct:.1f}%")
    col3.metric("Annual Cost Saving", f"£{annual_cost_saved:.2f}", delta=f"£{cost_saved:.2f}/route")
    col4.metric("Annual CO₂ Saved", f"{annual_co2_saved:.0f} kg")

    if cost_saved > 0 or co2_saved > 0:
        st.markdown(
            f"""
            <div class="judge-note">
                <strong>Judge takeaway:</strong> EcoRoute saves <strong>£{cost_saved:.2f} per route</strong> first, while also reducing estimated emissions by
                <strong>{co2_saved:.2f} kg CO₂ per route</strong> (<strong>{co2_reduction_pct:.1f}% reduction</strong>) for a <strong>{payload_kg:.0f} kg payload</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("In this scenario, EcoRoute did not outperform the standard route. Try changing stops, vehicle type, or payload weight.")

    # -----------------------------
    # Route comparison
    # -----------------------------
    st.markdown("## 🧭 Route Comparison")
    comparison_df = pd.DataFrame(
        [
            {
                "Metric": "Route sequence",
                "Standard Route": " → ".join(standard_route),
                "EcoRoute": " → ".join(eco_route),
            },
            {
                "Metric": "Road distance (km)",
                "Standard Route": standard_impact["road_distance_km"],
                "EcoRoute": eco_impact["road_distance_km"],
            },
            {
                "Metric": "Estimated duration (mins)",
                "Standard Route": standard_impact["duration_minutes"],
                "EcoRoute": eco_impact["duration_minutes"],
            },
            {
                "Metric": "Payload weight (kg)",
                "Standard Route": standard_impact["payload_kg"],
                "EcoRoute": eco_impact["payload_kg"],
            },
            {
                "Metric": "Payload multiplier",
                "Standard Route": standard_impact["payload_multiplier"],
                "EcoRoute": eco_impact["payload_multiplier"],
            },
            {
                "Metric": "Estimated cost (£)",
                "Standard Route": standard_impact["cost_gbp"],
                "EcoRoute": eco_impact["cost_gbp"],
            },
            {
                "Metric": "Rule-based CO₂ (kg)",
                "Standard Route": standard_impact["co2_kg"],
                "EcoRoute": eco_impact["co2_kg"],
            },
            {
                "Metric": "ML-predicted CO₂ (kg)",
                "Standard Route": round(standard_ml_co2, 2),
                "EcoRoute": round(eco_ml_co2, 2),
            },
        ]
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # -----------------------------
    # Map
    # -----------------------------
    st.markdown("## 🗺️ Route Map")
    st.write("**Red line:** Standard route | **Green dashed line:** EcoRoute | **Blue marker:** Depot")
    draw_route_map(
        standard_route=standard_route,
        eco_route=eco_route,
        standard_geometry=standard_result.get("geometry"),
        eco_geometry=eco_result.get("geometry"),
    )

    # -----------------------------
    # ML section
    # -----------------------------
    st.markdown("## 🤖 ML Emissions Prediction Layer")
    ml1, ml2, ml3 = st.columns(3)
    ml1.metric("ML Predicted Standard CO₂", f"{standard_ml_co2:.2f} kg")
    ml2.metric("ML Predicted EcoRoute CO₂", f"{eco_ml_co2:.2f} kg", delta=f"-{ml_co2_reduction_pct:.1f}%")
    ml3.metric("ML Predicted CO₂ Saving", f"{ml_co2_saved:.2f} kg")

    with st.expander("Why did the model make this recommendation?", expanded=False):
        st.markdown("**Standard route factors:**")
        for item in explain_route_factors(vehicle, payload_kg, "Standard"):
            st.write(f"- {item}")

        st.markdown("**EcoRoute factors:**")
        for item in explain_route_factors(vehicle, payload_kg, "EcoRoute"):
            st.write(f"- {item}")

        st.caption(
            "Rule-based values are deterministic calculations. ML predictions are learned estimates and may differ slightly. "
            "Mapbox handles traffic-aware routing; payload weight is supplied separately because mapping APIs do not know the vehicle load."
        )

    # -----------------------------
    # Route order breakdown
    # -----------------------------
    left, right = st.columns(2)
    with left:
        st.markdown("### Standard Route Order")
        st.dataframe(route_table_from_order(standard_route), use_container_width=True, hide_index=True)

    with right:
        st.markdown("### EcoRoute Order")
        st.dataframe(route_table_from_order(eco_route), use_container_width=True, hide_index=True)

    # -----------------------------
    # Judge-friendly takeaway
    # -----------------------------
    st.markdown("## Final Analysis")
    st.info(
        f"For a **{vehicle.lower()}** carrying an estimated **{payload_kg:.0f} kg payload**, "
        f"EcoRoute could save approximately **£{cost_saved:.2f} per route**, **£{annual_cost_saved:.2f} annually**, "
        f"and **{annual_co2_saved:.0f} kg CO₂ annually** for this recurring last-mile delivery pattern."
    )

else:
    st.info("Select vehicle type, payload weight, delivery stops, then click **Run EcoRoute Analysis** to generate the demo.")

    st.markdown("## 🎤 Suggested Live Pitch Flow")
    st.markdown(
        """
1. Add a Mapbox token if available to enable live traffic-aware routing.
2. Select vehicle type, delivery payload weight, and delivery stops.
3. Click **Run EcoRoute Analysis**.
4. Lead with annual cost saving.
5. Then show CO₂ reduction and route map.
6. Explain that Mapbox provides road-network routing, while EcoRoute adds business cost, payload and carbon impact logic.
        """
    )

    st.markdown("## Recommended Requirements")
    st.code(
        """
streamlit
pandas
requests
folium
streamlit-folium
scikit-learn
        """.strip(),
        language="text",
    )
