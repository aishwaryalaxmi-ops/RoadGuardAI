from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from datetime import datetime

import folium
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from ultralytics import YOLO
from streamlit_folium import st_folium
from streamlit_js_eval import streamlit_js_eval
ROOT = Path(__file__).resolve().parent

DB_PATH = ROOT / "data" / "roadguard.db"

VISION_MODEL_PATH = ROOT / "yolo11n.pt"
from src.model_utils import (
    DATA_PATH,
    load_bundle,
    load_metrics,
    predict,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

VISION_MODEL_PATH = ROOT / "yolo11n.pt"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 80% 10%,
                rgba(37, 99, 235, 0.08),
                transparent 30%
            ),
            #0b0f17;
    }

    .main {
        background: #0b0f17;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .brand {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: -5px;
        margin-bottom: 25px;
    }

    .hero {
        padding: 30px;
        border-radius: 22px;
        margin: 15px 0 25px 0;

        background:
            radial-gradient(
                circle at 85% 20%,
                rgba(59,130,246,0.22),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                rgba(15,23,42,0.98),
                rgba(30,41,59,0.85)
            );

        border: 1px solid rgba(96,165,250,0.18);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.6;
    }

    .status-online {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(34,197,94,0.12);
        color: #4ade80;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #f8fafc;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    .rg-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.15);

        background:
            linear-gradient(
                145deg,
                rgba(30,41,59,0.90),
                rgba(15,23,42,0.75)
            );

        min-height: 120px;
        margin-bottom: 15px;
    }

    .rg-card-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .rg-card-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .rg-card-description {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 5px;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BRAND HEADER
# ============================================================

st.markdown(
    '<div class="brand">🛡️ ROADGUARD AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Traffic Intelligence & Road Safety Prediction Platform'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(value: str) -> str:
    return str(value).strip().title()


def risk_icon(label: str) -> str:
    return {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🔴",
    }.get(
        normalize_label(label),
        "⚪",
    )


def load_data() -> pd.DataFrame | None:

    if not DATA_PATH.exists():
        return None

    try:
        return pd.read_csv(DATA_PATH)

    except Exception as exc:
        st.error(
            f"Dataset could not be read: {exc}"
        )

        return None


def risk_signals(p: dict) -> list[str]:

    signals = []

    if normalize_label(
        p["traffic_density"]
    ) == "High":
        signals.append(
            "High traffic density"
        )

    if float(
        p["average_speed"]
    ) >= 70:
        signals.append(
            "Elevated vehicle speed"
        )

    if normalize_label(
        p["weather"]
    ) in {
        "Rainy",
        "Stormy",
        "Foggy",
    }:
        signals.append(
            f"Reduced/unstable weather condition: "
            f"{p['weather']}"
        )

    if float(
        p["visibility"]
    ) < 5:
        signals.append(
            "Low visibility"
        )

    if normalize_label(
        p["road_condition"]
    ) in {
        "Wet",
        "Poor",
    }:
        signals.append(
            f"Road condition: "
            f"{p['road_condition']}"
        )

    if normalize_label(
        p["accident_history"]
    ) == "High":
        signals.append(
            "High historical accident category"
        )

    if normalize_label(
        p["traffic_violations"]
    ) == "High":
        signals.append(
            "High traffic-violation category"
        )

    if normalize_label(
        p["construction_zone"]
    ) == "Yes":
        signals.append(
            "Construction zone present"
        )

    if normalize_label(
        p["pedestrian_density"]
    ) == "High":
        signals.append(
            "High pedestrian density"
        )

    return signals[:6]


# ============================================================
# OPTIONAL YOLO VISION MODEL
# ============================================================

@st.cache_resource
def load_vision_model():

    try:

        from ultralytics import YOLO

        if VISION_MODEL_PATH.exists():

            return YOLO(
                str(VISION_MODEL_PATH)
            )

        # If model is not inside project,
        # YOLO can download it automatically.
        return YOLO("yolo11n.pt")

    except Exception:

        return None


# ============================================================
# VISION ANALYSIS
# ============================================================

def analyze_road_image(image_file):

    import numpy as np
    from PIL import Image

    model = load_vision_model()

    if model is None:

        raise RuntimeError(
            "YOLO is not available. "
            "Run: python -m pip install ultralytics"
        )

    image = Image.open(
        image_file
    ).convert("RGB")

    image_array = np.array(
        image
    )

    results = model(
        image_array,
        conf=0.35,
        verbose=False,
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = result.names[
                class_id
            ]

            detections.append(
                {
                    "object": class_name,
                    "confidence": confidence,
                }
            )

    annotated = result.plot()

    return detections, annotated

# ============================================================
# PROFESSIONAL DASHBOARD
# ============================================================

def dashboard_page():

    data = load_data()
    metrics = load_metrics()

    # ========================================================
    # HEADER
    # ========================================================

    st.success(
        "● SYSTEM ONLINE"
    )

    st.write(
        "Monitor traffic conditions, inspect road-risk patterns, "
        "analyze safety events and review AI-assisted road intelligence."
    )

    st.divider()

    # ========================================================
    # DATA STATUS
    # ========================================================

    dataset_rows = (
        len(data)
        if data is not None
        else 0
    )

    traffic_model = (
        metrics.get(
            "traffic",
            {}
        ).get(
            "model_name",
            "Not trained"
        )
        if metrics
        else "Not trained"
    )

    accident_model = (
        metrics.get(
            "accident",
            {}
        ).get(
            "model_name",
            "Not trained"
        )
        if metrics
        else "Not trained"
    )

    # ========================================================
    # TOP METRICS
    # ========================================================

    st.subheader(
        "Live Safety Overview"
    )

    traffic_status = "MONITORING"
    safety_status = "MONITORING"
    vehicle_total = 0
    active_alerts = 0

    if data is not None:

        # Vehicle count
        if "vehicle_count" in data.columns:

            vehicle_total = int(
                pd.to_numeric(
                    data["vehicle_count"],
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            )

        # Traffic status
        if "traffic_level" in data.columns:

            traffic_values = (
                data["traffic_level"]
                .astype(str)
                .str.strip()
                .str.title()
            )

            high_traffic = int(
                (traffic_values == "High").sum()
            )

            traffic_status = (
                "HIGH"
                if high_traffic > 0
                else "NORMAL"
            )

        # Accident alerts
        if "accident_risk" in data.columns:

            risk_values = (
                data["accident_risk"]
                .astype(str)
                .str.strip()
                .str.title()
            )

            active_alerts = int(
                (risk_values == "High").sum()
            )

            safety_status = (
                "HIGH RISK"
                if active_alerts > 0
                else "MONITORING"
            )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "🚦 Traffic",
            traffic_status
        )

    with m2:
        st.metric(
            "🛡️ Safety",
            safety_status
        )

    with m3:
        st.metric(
            "🚗 Vehicle Records",
            f"{vehicle_total:,}"
        )

    with m4:
        st.metric(
            "🚨 Risk Alerts",
            f"{active_alerts:,}"
        )

    st.divider()

    # ========================================================
    # MODEL STATUS
    # ========================================================

    st.subheader(
        "🤖 AI System Status"
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric(
            "Dataset",
            f"{dataset_rows:,}"
        )

    with a2:
        st.metric(
            "Traffic AI",
            traffic_model
        )

    with a3:
        st.metric(
            "Accident AI",
            accident_model
        )

    with a4:
        st.metric(
            "System",
            "READY" if metrics else "NOT READY"
        )

    st.divider()

    # ========================================================
    # MAIN COMMAND CENTER
    # ========================================================

    left, right = st.columns([1.35, 1])

    # ========================================================
    # RISK MAP
    # ========================================================

    with left:

        st.subheader(
            "🗺️ Live Risk Map"
        )

        if (
            data is not None
            and {
                "latitude",
                "longitude"
            }.issubset(data.columns)
        ):

            map_data = data.dropna(
                subset=[
                    "latitude",
                    "longitude"
                ]
            )

            if not map_data.empty:

                center_lat = map_data[
                    "latitude"
                ].mean()

                center_lon = map_data[
                    "longitude"
                ].mean()

                dashboard_map = folium.Map(
                    location=[
                        center_lat,
                        center_lon
                    ],
                    zoom_start=11,
                    control_scale=True
                )

                risk_colors = {
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red"
                }

                for _, row in map_data.head(250).iterrows():

                    risk = normalize_label(
                        row.get(
                            "accident_risk",
                            "Low"
                        )
                    )

                    popup = (
                        f"Location: "
                        f"{row.get('location', 'Road Segment')}"
                        f"<br>Risk: {risk}"
                        f"<br>Traffic: "
                        f"{row.get('traffic_level', 'N/A')}"
                    )

                    folium.CircleMarker(
                        location=[
                            row["latitude"],
                            row["longitude"]
                        ],
                        radius=5,
                        color=risk_colors.get(
                            risk,
                            "blue"
                        ),
                        fill=True,
                        fill_opacity=0.75,
                        popup=popup
                    ).add_to(dashboard_map)

                st_folium(
                    dashboard_map,
                    use_container_width=True,
                    height=480
                )

            else:

                st.info(
                    "No valid GPS records are available."
                )

        else:

            st.info(
                "Latitude and longitude data are required "
                "for the live risk map."
            )

    # ========================================================
    # TRAFFIC + ACCIDENT ANALYTICS
    # ========================================================

    with right:

        st.subheader(
            "📊 Risk Intelligence"
        )

        if (
            data is not None
            and "traffic_level" in data.columns
        ):

            traffic_chart = px.histogram(
                data,
                x="traffic_level",
                category_orders={
                    "traffic_level": [
                        "Low",
                        "Medium",
                        "High"
                    ]
                },
                title="Traffic Conditions"
            )

            traffic_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220
            )

            st.plotly_chart(
                traffic_chart,
                use_container_width=True
            )

        if (
            data is not None
            and "accident_risk" in data.columns
        ):

            accident_chart = px.histogram(
                data,
                x="accident_risk",
                category_orders={
                    "accident_risk": [
                        "Low",
                        "Medium",
                        "High"
                    ]
                },
                title="Accident Risk"
            )

            accident_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220
            )

            st.plotly_chart(
                accident_chart,
                use_container_width=True
            )

    st.divider()

    # ========================================================
    # ROADGUARD MODULES
    # ========================================================

    st.subheader(
        "🚘 RoadGuard Intelligence Modules"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "🚦 Traffic Intelligence",
            "📷 AI Road Vision",
            "🚨 Safety Events"
        ]
    )

    # ========================================================
    # TRAFFIC TAB
    # ========================================================

    with tab1:

        if (
            data is not None
            and {
                "traffic_volume",
                "average_speed"
            }.issubset(data.columns)
        ):

            traffic_scatter = px.scatter(
                data,
                x="traffic_volume",
                y="average_speed",
                color=(
                    "traffic_level"
                    if "traffic_level"
                    in data.columns
                    else None
                ),
                title="Traffic Volume vs Average Speed"
            )

            traffic_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                traffic_scatter,
                use_container_width=True
            )

        else:

            st.info(
                "Traffic volume and speed data are not available."
            )

    # ========================================================
    # AI VISION TAB
    # ========================================================

    with tab2:

        st.info(
            """
            ### 📷 AI Road Vision

            Capture a road scene and send it to the
            RoadGuard computer-vision engine.

            **Available through the Road Camera and AI Vision pages.**
            """
        )

        

    # ========================================================
    # SAFETY EVENTS TAB
    # ========================================================

    with tab3:

        if (
            data is not None
            and "accident_risk" in data.columns
        ):

            high_risk_data = data[
                data["accident_risk"]
                .astype(str)
                .str.strip()
                .str.title()
                == "High"
            ]

            if not high_risk_data.empty:

                display_columns = [
                    column
                    for column in [
                        "location",
                        "traffic_level",
                        "weather",
                        "accident_risk"
                    ]
                    if column in high_risk_data.columns
                ]

                st.dataframe(
                    high_risk_data[
                        display_columns
                    ].head(10),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "No high-risk records found in the dataset."
                )

        else:

            st.info(
                "Accident-risk data is not available."
            )

    st.divider()

    # ========================================================
    # QUICK ACTIONS
    # ========================================================

    st.subheader(
        "⚡ Quick Actions"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.info(
            "🚨 **Safety Scan**\n\n"
            "Evaluate current road conditions."
        )

    with q2:
        st.info(
            "📷 **Road Camera**\n\n"
            "Capture a road scene."
        )

    with q3:
        st.info(
            "🗺️ **Risk Map**\n\n"
            "Explore geographic risk."
        )

    with q4:
        st.info(
            "📊 **Analytics**\n\n"
            "Review traffic and accident patterns."
        )

    # ========================================================
    # DATA WARNING
    # ========================================================

    if data is None:

        st.warning(
            "Dataset not available. Add "
            "traffic_accident_data.csv to the data folder."
        )

# ============================================================
# RISK PREDICTION PAGE
# ============================================================

def prediction_page():

    st.header(
        "🧠 Risk Prediction"
    )

    st.caption(
        "Enter road conditions and receive separate "
        "traffic and accident-risk model estimates."
    )

    with st.form("risk_form"):

        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "📍 Basic Information"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            date = st.date_input(
                "Date"
            )

        with c2:

            time = st.time_input(
                "Time"
            )

        with c3:

            location = st.text_input(
                "Location",
                "Bhubaneswar"
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            latitude = st.number_input(
                "Latitude",
                value=20.2961,
                format="%.6f"
            )

        with c2:

            longitude = st.number_input(
                "Longitude",
                value=85.8245,
                format="%.6f"
            )

        with c3:

            traffic_volume = st.number_input(
                "Traffic volume",
                min_value=0,
                value=500,
                step=10
            )

        # ----------------------------------------------------
        # TRAFFIC
        # ----------------------------------------------------

        st.subheader(
            "🚦 Traffic Conditions"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            traffic_density = st.selectbox(
                "Traffic density",
                [
                    "Low",
                    "Medium",
                    "High",
                ]
            )

        with c2:

            vehicle_count = st.number_input(
                "Vehicle count",
                min_value=0,
                value=550,
                step=10
            )

        with c3:

            average_speed = st.number_input(
                "Average speed (km/h)",
                min_value=0.0,
                value=45.0,
                step=1.0
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            queue_length = st.number_input(
                "Queue length",
                min_value=0,
                value=12,
                step=1
            )

        with c2:

            lane_occupancy = st.slider(
                "Lane occupancy",
                0.0,
                1.0,
                0.50,
                0.01
            )

        with c3:

            weather = st.selectbox(
                "Weather",
                [
                    "Clear",
                    "Cloudy",
                    "Rainy",
                    "Foggy",
                    "Stormy",
                ]
            )

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        st.subheader(
            "🌦️ Environment"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            temperature = st.number_input(
                "Temperature (°C)",
                value=28.0,
                step=0.5
            )

        with c2:

            rainfall = st.number_input(
                "Rainfall",
                min_value=0.0,
                value=0.0,
                step=0.5
            )

        with c3:

            visibility = st.number_input(
                "Visibility (km)",
                min_value=0.1,
                value=10.0,
                step=0.5
            )

        # ----------------------------------------------------
        # ROAD
        # ----------------------------------------------------

        st.subheader(
            "🛣️ Road Conditions"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            road_condition = st.selectbox(
                "Road condition",
                [
                    "Good",
                    "Wet",
                    "Poor",
                ]
            )

        with c2:

            road_type = st.selectbox(
                "Road type",
                [
                    "Urban",
                    "Highway",
                    "Residential",
                    "Arterial",
                ]
            )

        with c3:

            road_width = st.number_input(
                "Road width (m)",
                min_value=1.0,
                value=10.0,
                step=0.5
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            number_of_lanes = st.number_input(
                "Number of lanes",
                min_value=1,
                value=2,
                step=1
            )

        with c2:

            intersection_type = st.selectbox(
                "Intersection type",
                [
                    "Signalized",
                    "Roundabout",
                    "T-Junction",
                    "Crossroad",
                ]
            )

        with c3:

            construction_zone = st.selectbox(
                "Construction zone",
                [
                    "No",
                    "Yes",
                ]
            )

        # ----------------------------------------------------
        # ACCIDENT HISTORY
        # ----------------------------------------------------

        st.subheader(
            "⚠️ Accident History"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            accident_history = st.selectbox(
                "Accident history",
                [
                    "Low",
                    "Medium",
                    "High",
                ]
            )

        with c2:

            traffic_violations = st.selectbox(
                "Traffic violations",
                [
                    "Low",
                    "Medium",
                    "High",
                ]
            )

        with c3:

            pedestrian_density = st.selectbox(
                "Pedestrian density",
                [
                    "Low",
                    "Medium",
                    "High",
                ]
            )

        c1, c2 = st.columns(2)

        with c1:

            accident_count = st.number_input(
                "Recent accident count",
                min_value=0,
                value=2,
                step=1
            )

        with c2:

            accident_severity = st.selectbox(
                "Historical accident severity",
                [
                    "None",
                    "Minor",
                    "Moderate",
                    "Severe",
                ]
            )

        submitted = st.form_submit_button(
            "🔎 ANALYZE ROAD RISK",
            use_container_width=True
        )

    if not submitted:
        return

    # --------------------------------------------------------
    # PAYLOAD
    # --------------------------------------------------------

    payload = {

        "date": str(date),

        "time": time.strftime("%H:%M"),

        "latitude": latitude,

        "longitude": longitude,

        "location": location,

        "traffic_volume": traffic_volume,

        "traffic_density": traffic_density,

        "vehicle_count": vehicle_count,

        "average_speed": average_speed,

        "queue_length": queue_length,

        "lane_occupancy": lane_occupancy,

        "weather": weather,

        "temperature": temperature,

        "rainfall": rainfall,

        "visibility": visibility,

        "road_condition": road_condition,

        "road_type": road_type,

        "road_width": road_width,

        "number_of_lanes": number_of_lanes,

        "intersection_type": intersection_type,

        "construction_zone": construction_zone,

        "accident_history": accident_history,

        "traffic_violations": traffic_violations,

        "pedestrian_density": pedestrian_density,

        "accident_count": accident_count,

        "accident_severity": accident_severity,
    }

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        traffic_bundle = load_bundle(
            "traffic_model.pkl"
        )

        accident_bundle = load_bundle(
            "accident_model.pkl"
        )

        traffic_result = predict(
            traffic_bundle,
            payload
        )

        accident_result = predict(
            accident_bundle,
            payload
        )

    except Exception as exc:

        st.error(
            f"Prediction is unavailable: {exc}"
        )

        return

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.divider()

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Traffic Condition",
            f"{risk_icon(traffic_result['prediction'])} "
            f"{traffic_result['prediction']}"
        )

    with r2:

        st.metric(
            "Accident Risk",
            f"{risk_icon(accident_result['prediction'])} "
            f"{accident_result['prediction']}"
        )

    with r3:

        probability = accident_result.get(
            "probability"
        )

        if probability is not None:

            st.metric(
                "Model Probability",
                f"{probability * 100:.1f}%"
            )

        else:

            st.metric(
                "Model Probability",
                "N/A"
            )

    # --------------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------------

    st.subheader(
        "📊 Risk Probability Distribution"
    )

    probabilities = accident_result.get(
        "probabilities",
        {}
    )

    if probabilities:

        probs = pd.DataFrame(
            {
                "Risk": list(
                    probabilities.keys()
                ),

                "Model score": list(
                    probabilities.values()
                ),
            }
        )

        probs["Model score"] *= 100

        st.bar_chart(
            probs.set_index("Risk"),
            y="Model score"
        )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    st.subheader(
        "🔍 Observed Risk Signals"
    )

    signals = risk_signals(
        payload
    )

    if signals:

        for signal in signals:

            st.write(
                f"• {signal}"
            )

    else:

        st.write(
            "No strong rule-based warning signal "
            "was detected from the entered fields."
        )

    st.info(
        "The prediction is a model estimate for "
        "decision support, not a guarantee that an "
        "accident will or will not occur."
    )


# ============================================================
# SAFETY SCAN PAGE
# ============================================================

def safety_scan_page():
    st.header("👁️ RoadGuard AI Vision")

    st.write("Upload an image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="Uploaded road image",
            use_container_width=True
        )
    st.caption(
        "Perform a quick rule-based safety assessment "
        "from current road conditions."
    )

    st.markdown(
        """
        

            
                ● SAFETY ENGINE READY
            
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "Road Conditions"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        traffic = st.selectbox(
            "Traffic Level",
            [
                "Low",
                "Medium",
                "High",
            ]
        )

    with c2:

        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Cloudy",
                "Rainy",
                "Foggy",
                "Stormy",
            ]
        )

    with c3:

        road = st.selectbox(
            "Road Condition",
            [
                "Good",
                "Wet",
                "Poor",
            ]
        )

    visibility = st.slider(
        "Visibility (km)",
        0.1,
        20.0,
        10.0,
        0.1
    )

    if st.button(
        "🔎 RUN SAFETY SCAN",
        use_container_width=True,
        type="primary",
    ):

        risk_score = 0

        if traffic == "High":
            risk_score += 30

        elif traffic == "Medium":
            risk_score += 15

        if weather in [
            "Rainy",
            "Foggy",
            "Stormy",
        ]:
            risk_score += 25

        if road == "Wet":
            risk_score += 15

        elif road == "Poor":
            risk_score += 25

        if visibility < 5:
            risk_score += 20

        elif visibility < 8:
            risk_score += 10

        risk_score = min(
            risk_score,
            100
        )

        st.divider()

        st.subheader(
            "Safety Assessment"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Safety Risk Score",
                f"{risk_score}/100"
            )

        with c2:

            if risk_score >= 70:

                result = "🔴 HIGH"

            elif risk_score >= 40:

                result = "🟡 MEDIUM"

            else:

                result = "🟢 LOW"

            st.metric(
                "Current Risk",
                result
            )

        if risk_score >= 70:

            st.error(
                "High-risk conditions were detected "
                "from the entered inputs."
            )

        elif risk_score >= 40:

            st.warning(
                "Moderate-risk conditions were detected."
            )

        else:

            st.success(
                "No strong warning was detected "
                "from the entered conditions."
            )
def get_gps_location():
    location = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude,
                        accuracy: position.coords.accuracy
                    });
                },
                () => {
                    resolve(null);
                }
            );
        })
        """,
        key="gps_location"
    )

    return location

# ============================================================
# ROAD CAMERA + AI VISION
# ============================================================

@st.cache_resource
def load_vision_model():
    return YOLO("yolo11n.pt")


def get_gps_location():
    location = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude,
                        accuracy: position.coords.accuracy
                    });
                },
                () => {
                    resolve(null);
                }
            );
        })
        """,
        key="gps_location"
    )

    return location


def calculate_camera_risk(detections):

    vehicle_types = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle"
    }

    vehicle_count = sum(
        1
        for item in detections
        if item["object"].lower() in vehicle_types
    )

    pedestrian_count = sum(
        1
        for item in detections
        if item["object"].lower() == "person"
    )

    risk_score = 0
    reasons = []

    # Vehicle density
    if vehicle_count >= 10:
        risk_score += 50
        reasons.append(
            "High number of detected vehicles"
        )

    elif vehicle_count >= 5:
        risk_score += 25
        reasons.append(
            "Moderate vehicle density"
        )

    # Pedestrian density
    if pedestrian_count >= 8:
        risk_score += 30
        reasons.append(
            "High pedestrian presence"
        )

    elif pedestrian_count >= 4:
        risk_score += 15
        reasons.append(
            "Moderate pedestrian presence"
        )

    # Risk level
    if risk_score >= 60:
        risk_level = "HIGH"

    elif risk_score >= 30:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "risk_score": min(risk_score, 100),
        "vehicle_count": vehicle_count,
        "pedestrian_count": pedestrian_count,
        "reasons": reasons
    }


def analyze_road_image(image):

    model = load_vision_model()

    results = model.predict(
        image,
        conf=0.35,
        verbose=False
    )

    result = results[0]

    annotated_image = result.plot()

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[
                class_id
            ]

            detections.append(
                {
                    "object": class_name,
                    "confidence": confidence
                }
            )

    return detections, annotated_image


# ============================================================
# ROAD CAMERA PAGE
# ============================================================

def camera_page():

    image = st.camera_input("Take a picture")

    if image is None:
        return

    st.header("📷 Road Camera")

    st.caption(
        "Capture a road image using your device camera "
        "and prepare it for AI-powered safety analysis."
    )

    st.markdown(
        """
        

            
                ● CAMERA ENGINE READY
    
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # LOCATION
    # ========================================================

    st.subheader("📍 Location")

    if st.button(
        "📍 Get Current Location",
        use_container_width=True
    ):

        location = get_gps_location()

        if location:

            st.session_state[
                "gps_location"
            ] = location

            st.success(
                "Location captured successfully."
            )

        else:

            st.warning(
                "Location permission was not granted "
                "or GPS information is unavailable."
            )

    # Show saved location
    if "gps_location" in st.session_state:

        location = st.session_state[
            "gps_location"
        ]

        l1, l2, l3 = st.columns(3)

        with l1:
            st.metric(
                "Latitude",
                f"{location['latitude']:.6f}"
            )

        with l2:
            st.metric(
                "Longitude",
                f"{location['longitude']:.6f}"
            )

        with l3:
            st.metric(
                "Accuracy",
                f"±{location['accuracy']:.1f} m"
            )

    # ========================================================
    # CAMERA
    # ========================================================

    st.subheader("📸 Camera")

    image = st.camera_input(
        "Take a road picture"
    )

    if image is None:

        st.info(
            "Allow camera access and capture "
            "a road image to begin."
        )

        return

    st.success(
        "✅ Camera image captured successfully."
    )

    st.image(
        image,
        caption="Captured Road Scene",
        use_container_width=True
    )

    # ========================================================
    # AI ANALYSIS BUTTON
    # ========================================================

    if st.button(
        "🧠 ANALYZE ROAD SCENE",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner(
            "RoadGuard AI is analyzing the road scene..."
        ):

            try:

                detections, annotated = (
                    analyze_road_image(image)
                )

            except Exception as exc:

                st.error(
                    f"AI vision analysis failed: {exc}"
                )

                st.info(
                    "Install the required packages with:\n\n"
                    "python -m pip install ultralytics opencv-python"
                )

                return

        # ====================================================
        # DETECTION RESULTS
        # ====================================================

        st.divider()

        st.subheader(
            "🔍 Detection Results"
        )

        if not detections:

            st.info(
                "No supported objects were detected "
                "at the current confidence threshold."
            )

            return

        detection_df = pd.DataFrame(
            detections
        )

        detection_df[
            "confidence"
        ] = (
            detection_df[
                "confidence"
            ] * 100
        ).round(1)

        detection_df = detection_df.rename(
            columns={
                "object": "Detected Object",
                "confidence": "Confidence (%)"
            }
        )

        st.success(
            f"AI detected {len(detections)} object(s)."
        )

        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # OBJECT SUMMARY
        # ====================================================

        st.subheader(
            "📊 Object Summary"
        )

        counts = (
            detection_df[
                "Detected Object"
            ]
            .value_counts()
        )

        summary_columns = st.columns(
            min(len(counts), 4)
        )

        for index, (
            object_name,
            count
        ) in enumerate(
            counts.items()
        ):

            summary_columns[
                index % len(summary_columns)
            ].metric(
                object_name.title(),
                int(count)
            )

        # ====================================================
        # SAFETY ASSESSMENT
        # ====================================================

        safety = calculate_camera_risk(
            detections
        )

        st.subheader(
            "🚨 Road Safety Assessment"
        )

        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "Safety Risk",
                safety["risk_level"]
            )

        with s2:

            st.metric(
                "Risk Score",
                f'{safety["risk_score"]}/100'
            )

        with s3:

            st.metric(
                "Vehicles",
                safety["vehicle_count"]
            )

        st.metric(
            "Pedestrians",
            safety["pedestrian_count"]
        )

        # Risk message
        if safety["risk_level"] == "HIGH":

            st.error(
                "🔴 HIGH RISK: Immediate attention recommended."
            )

        elif safety["risk_level"] == "MEDIUM":

            st.warning(
                "🟡 MEDIUM RISK: Increased monitoring recommended."
            )

        else:

            st.success(
                "🟢 LOW RISK: No strong camera-based warning detected."
            )

        # ====================================================
        # SAFETY SIGNALS
        # ====================================================

        if safety["reasons"]:

            st.subheader(
                "⚠️ Detected Safety Signals"
            )

            for reason in safety["reasons"]:

                st.write(
                    f"• {reason}"
                )

        # ====================================================
        # GPS SAFETY EVENT
        # ====================================================

        if "gps_location" in st.session_state:

            location = st.session_state[
                "gps_location"
            ]

            st.subheader(
                "📍 Safety Event Location"
            )

            g1, g2, g3 = st.columns(3)

            with g1:

                st.metric(
                    "Latitude",
                    f"{location['latitude']:.6f}"
                )

            with g2:

                st.metric(
                    "Longitude",
                    f"{location['longitude']:.6f}"
                )

            with g3:

                st.metric(
                    "GPS Accuracy",
                    f"±{location['accuracy']:.1f} m"
                )

        # ====================================================
        # ANNOTATED IMAGE
        # ====================================================

        st.subheader(
            "🧠 AI Vision Output"
        )

        st.image(
            annotated,
            caption="RoadGuard AI Detected Objects",
            use_container_width=True
        )


# ============================================================
# AI VISION PAGE
# ============================================================

def vision_page():

    st.header(
        "👁️ AI Vision"
    )

    st.caption(
        "Use computer vision to detect visible "
        "objects in a road scene."
    )

    st.markdown(
        """
        

            
                ● VISION ENGINE READY
            
        """,
        unsafe_allow_html=True
    )

    image = st.camera_input(
        "📷 Take a picture for AI analysis"
    )

    if image is None:

        st.info(
            "Allow camera access and capture "
            "a road image to begin."
        )

        return

    st.success(
        "✅ Image captured successfully."
    )

    if st.button(
        "🧠 ANALYZE ROAD SCENE",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner(
            "RoadGuard Vision is analyzing the scene..."
        ):

            try:

                detections, annotated = (
                    analyze_road_image(image)
                )

            except Exception as exc:

                st.error(
                    f"AI vision analysis failed: {exc}"
                )

                return

        st.divider()

        st.subheader(
            "🔍 Detection Results"
        )

        if not detections:

            st.info(
                "No supported objects were detected."
            )

            return

        detection_df = pd.DataFrame(
            detections
        )

        detection_df[
            "confidence"
        ] = (
            detection_df[
                "confidence"
            ] * 100
        ).round(1)

        detection_df = detection_df.rename(
            columns={
                "object": "Detected Object",
                "confidence": "Confidence (%)"
            }
        )

        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "📊 Object Summary"
        )

        counts = (
            detection_df[
                "Detected Object"
            ]
            .value_counts()
        )

        summary_columns = st.columns(
            min(len(counts), 4)
        )

        for index, (
            object_name,
            count
        ) in enumerate(
            counts.items()
        ):

            summary_columns[
                index % len(summary_columns)
            ].metric(
                object_name.title(),
                int(count)
            )

        st.subheader(
            "🧠 AI Vision Output"
        )

        st.image(
            annotated,
            caption="RoadGuard AI Detected Objects",
            use_container_width=True
        )

        # ----------------------------------------------------
        # SAFETY INDICATORS
        # ----------------------------------------------------

        vehicle_classes = {
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
        }

        people_count = int(
            counts.get(
                "person",
                0
            )
        )

        vehicle_count = sum(
            int(
                counts.get(
                    name,
                    0
                )
            )
            for name in vehicle_classes
        )

        st.subheader(
            "🚨 Vision Safety Indicators"
        )

        v1, v2, v3 = st.columns(3)

        with v1:

            st.metric(
                "Vehicles Detected",
                vehicle_count
            )

        with v2:

            st.metric(
                "People Detected",
                people_count
            )

        with v3:

            st.metric(
                "Total Objects",
                len(detections)
            )

        if vehicle_count >= 10:

            st.warning(
                "🟡 High number of visible vehicles "
                "detected in this image."
            )

        elif vehicle_count >= 5:

            st.info(
                "🔵 Moderate vehicle activity "
                "detected."
            )

        else:

            st.success(
                "🟢 Lower visible vehicle activity "
                "detected."
            )

        st.caption(
            "Vision results are based only on visible "
            "objects in the captured image."
        )


#============================================================
#GPS SAFETY PAGE
# ============================================================

def gps_page():

    st.header(
        "📍 GPS Safety"
    )

    st.caption(
        "Enter or use a known location for "
        "location-aware road analysis."
    )

    st.subheader(
        "Current Location"
    )

    c1, c2 = st.columns(2)

    with c1:

        latitude = st.number_input(
            "Latitude",
            value=20.2961,
            format="%.6f"
        )

    with c2:

        longitude = st.number_input(
            "Longitude",
            value=85.8245,
            format="%.6f"
        )

    location_name = st.text_input(
        "Location Name",
        "Bhubaneswar"
    )

    st.success(
        f"📍 {location_name}"
    )

    st.write(
        f"Latitude: `{latitude}`"
    )

    st.write(
        f"Longitude: `{longitude}`"
    )

    m = folium.Map(
        location=[
            latitude,
            longitude,
        ],
        zoom_start=14,
    )

    folium.Marker(
        [
            latitude,
            longitude,
        ],
        popup=location_name,
        tooltip="Current Location",
        icon=folium.Icon(
            color="blue",
            icon="info-sign"
        ),
    ).add_to(m)

    st_folium(
        m,
        use_container_width=True,
        height=500,
    )
# ============================================================
# SAFETY EVENT DATABASE
# ============================================================

def init_database():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                accuracy REAL,
                risk_level TEXT,
                risk_score INTEGER,
                vehicles INTEGER,
                pedestrians INTEGER
            )
            """
        )

        conn.commit()

init_database()
# ============================================================
# CAMERA EVENT MAP
# ============================================================

def save_safety_event(safety, location):

    event = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "accuracy": location["accuracy"],
        "risk_level": safety["risk_level"],
        "risk_score": safety["risk_score"],
        "vehicles": safety["vehicle_count"],
        "pedestrians": safety["pedestrian_count"],
    }

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            INSERT INTO safety_events (
                timestamp,
                latitude,
                longitude,
                accuracy,
                risk_level,
                risk_score,
                vehicles,
                pedestrians
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["timestamp"],
                event["latitude"],
                event["longitude"],
                event["accuracy"],
                event["risk_level"],
                event["risk_score"],
                event["vehicles"],
                event["pedestrians"],
            )
        )

        conn.commit()
def load_safety_events():

    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        df = pd.read_sql_query(
            """
            SELECT
                id,
                timestamp,
                latitude,
                longitude,
                accuracy,
                risk_level,
                risk_score,
                vehicles,
                pedestrians
            FROM safety_events
            ORDER BY id DESC
            """,
            conn
        )

    return df

# ============================================================
# RISK MAP
# ============================================================

def map_page():

    st.header("🗺️ RoadGuard Risk Map")

    st.caption(
        "View detected road-risk events and "
        "road-risk information geographically."
    )

    data = load_data()

    # --------------------------------------------------------
    # DEFAULT MAP LOCATION
    # --------------------------------------------------------

    default_lat = 20.2961
    default_lon = 85.8245

    risk_colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }

    if (
        data is not None
        and {"latitude", "longitude"}.issubset(data.columns)
    ):

        clean_data = data.dropna(
            subset=["latitude", "longitude"]
        )

        if not clean_data.empty:

            default_lat = clean_data[
                "latitude"
            ].mean()

            default_lon = clean_data[
                "longitude"
            ].mean()

    # --------------------------------------------------------
    # CREATE MAP
    # --------------------------------------------------------

    road_map = folium.Map(
        location=[
            default_lat,
            default_lon
        ],
        zoom_start=12,
        control_scale=True
    )

    # --------------------------------------------------------
    # DATASET RISK MARKERS
    # --------------------------------------------------------

    if data is not None:

        if {
            "latitude",
            "longitude"
        }.issubset(data.columns):

            sample = (
                data
                .dropna(
                    subset=[
                        "latitude",
                        "longitude"
                    ]
                )
                .head(500)
            )


            for _, row in sample.iterrows():

                risk = normalize_label(
                    row.get(
                        "accident_risk",
                        "Low"
                    )
                )

                popup = f"""
                <b>RoadGuard AI</b><br>
                Location: {row.get('location', 'Road segment')}<br>
                Risk: {risk}<br>
                Traffic: {row.get('traffic_level', 'N/A')}
                """

                folium.CircleMarker(
                    location=[
                        row["latitude"],
                        row["longitude"]
                    ],
                    radius=6,
                    color=risk_colors.get(
                        risk,
                        "blue"
                    ),
                    fill=True,
                    fill_opacity=0.75,
                    popup=popup
                ).add_to(road_map)

    # --------------------------------------------------------
    # CAMERA SAFETY EVENTS
    # --------------------------------------------------------

    camera_events_df = load_safety_events()

    for _, event in camera_events_df.iterrows():

        risk = str(
            event["risk_level"]
        )

        popup = f"""
        <b>🚨 RoadGuard AI Camera Event</b><br><br>

        Risk Level: {risk}<br>
        Risk Score: {int(event["risk_score"])}/100<br>
        Vehicles: {int(event["vehicles"])}<br>
        Pedestrians: {int(event["pedestrians"])}<br>
        GPS Accuracy: ±{float(event["accuracy"]):.1f} m<br>
        Time: {event["timestamp"]}
        """

        folium.Marker(
            location=[
                float(event["latitude"]),
                float(event["longitude"])
            ],
            popup=folium.Popup(
                popup,
                max_width=300
            ),
            tooltip=f"RoadGuard {risk} Risk",
            icon=folium.Icon(
                color=risk_colors.get(
                    risk,
                    "blue"
                ),
                icon="warning-sign"
            )
        ).add_to(road_map)

    # --------------------------------------------------------
    # MAP DISPLAY
    # --------------------------------------------------------

    st_folium(
        road_map,
        use_container_width=True,
        height=650
    )

    st.subheader(
        "📜 Safety Event History"
    )

    if not camera_events_df.empty:

        event_df = camera_events_df.copy()

    else:

        history_display = camera_events_df.copy()

        history_display = history_display.rename(
            columns={
                "id": "ID",
                "timestamp": "Time",
                "latitude": "Latitude",
                "longitude": "Longitude",
                "accuracy": "GPS Accuracy",
                "risk_level": "Risk",
                "risk_score": "Risk Score",
                "vehicles": "Vehicles",
                "pedestrians": "Pedestrians",
            }
        )

        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # CAMERA EVENT SUMMARY
        # --------------------------------------------------------

        st.subheader(
            "🚨 Camera Safety Events"
        )

        if not camera_events_df.empty:

            event_df = camera_events_df.copy()

            event_df = event_df.rename(
                columns={
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "risk_level": "Risk Level",
                    "risk_score": "Risk Score",
                    "vehicles": "Vehicles",
                    "pedestrians": "Pedestrians",
                    "accuracy": "GPS Accuracy"
                }
            )

            st.dataframe(
                event_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No camera safety events have been "
                "recorded during this session."
            )

# ANALYTICS PAGE
# ============================================================

def analytics_page():

    st.header(
        "📊 Analytics"
    )

    data = load_data()

    if data is None:

        st.warning(
            "Dataset not available."
        )

        return

    # --------------------------------------------------------
    # TRAFFIC VOLUME VS SPEED
    # --------------------------------------------------------

    if {
        "traffic_volume",
        "average_speed",
    }.issubset(
        data.columns
    ):

        fig = px.scatter(
            data,
            x="traffic_volume",
            y="average_speed",
            color=(
                "traffic_level"
                if "traffic_level"
                in data.columns
                else None
            ),
            title=(
                "Traffic Volume vs Average Speed"
            ),
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # WEATHER VS ACCIDENTS
    # --------------------------------------------------------

    if {
        "weather",
        "accident_count",
    }.issubset(
        data.columns
    ):

        weather_df = (
            data.groupby(
                "weather",
                as_index=False
            )[
                "accident_count"
            ]
            .mean()
            .sort_values(
                "accident_count",
                ascending=False
            )
        )

        fig = px.bar(
            weather_df,
            x="weather",
            y="accident_count",
            title=(
                "Mean Accident Count by Weather"
            ),
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # ROAD CONDITIONS
    # --------------------------------------------------------

    if {
        "road_condition",
        "accident_count",
    }.issubset(
        data.columns
    ):

        road_df = (
            data.groupby(
                "road_condition",
                as_index=False
            )[
                "accident_count"
            ]
            .mean()
        )

        fig = px.bar(
            road_df,
            x="road_condition",
            y="accident_count",
            title=(
                "Mean Accident Count by Road Condition"
            ),
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def performance_page():

    st.header(
        "🧠 Model Performance"
    )

    metrics = load_metrics()

    if not metrics:

        st.warning(
            "Run `python src/train_models.py` first."
        )

        return

    for key, title in [
        (
            "traffic",
            "Traffic Model"
        ),
        (
            "accident",
            "Accident-Risk Model"
        ),
    ]:

        if key not in metrics:
            continue

        info = metrics[key]

        st.subheader(
            title
        )

        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )

        m = info["metrics"]

        c1.metric(
            "Accuracy",
            f"{m['accuracy'] * 100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{m['precision_macro'] * 100:.2f}%"
        )

        c3.metric(
            "Recall",
            f"{m['recall_macro'] * 100:.2f}%"
        )

        c4.metric(
            "Macro F1",
            f"{m['f1_macro'] * 100:.2f}%"
        )

        auc = m.get(
            "roc_auc_ovr_weighted"
        )

        c5.metric(
            "ROC-AUC",
            (
                f"{auc:.3f}"
                if auc is not None
                else "N/A"
            )
        )

        # ----------------------------------------------------
        # LEADERBOARD
        # ----------------------------------------------------

        if "cv_leaderboard" in info:

            leaderboard = (
                pd.DataFrame(
                    info[
                        "cv_leaderboard"
                    ]
                )
                .T
                .reset_index()
                .rename(
                    columns={
                        "index": "Model"
                    }
                )
            )

            if {
                "cv_f1_macro_mean",
                "cv_f1_macro_std",
            }.issubset(
                leaderboard.columns
            ):

                leaderboard[
                    "CV Macro-F1"
                ] = (
                    leaderboard[
                        "cv_f1_macro_mean"
                    ]
                    .round(4)
                )

                st.dataframe(
                    leaderboard[
                        [
                            "Model",
                            "CV Macro-F1",
                            "cv_f1_macro_std",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        if "confusion_matrix" in m:

            cm = pd.DataFrame(
                m[
                    "confusion_matrix"
                ],
                index=[
                    "Actual Low",
                    "Actual Medium",
                    "Actual High",
                ],
                columns=[
                    "Pred Low",
                    "Pred Medium",
                    "Pred High",
                ],
            )

            st.dataframe(
                cm,
                use_container_width=True,
            )

        st.caption(
            "These values are generated from the "
            "dataset used during the latest training run. "
            "They are not universal accuracy claims."
        )
# ============================================================
# ALERTS & INCIDENTS PAGE
# ============================================================

def alerts_page():

    st.header("🚨 Alerts & Incidents")

    st.caption(
        "Review recorded road-safety events and their risk levels."
    )

    camera_events_df = load_safety_events()

    if camera_events_df.empty:

        st.success(
            "✅ No safety incidents have been recorded yet."
        )

        return

    # --------------------------------------------------------
    # ALERT COUNTS
    # --------------------------------------------------------

    high_count = int(
        (
            camera_events_df["risk_level"]
            .astype(str)
            .str.upper()
            == "HIGH"
        ).sum()
    )

    medium_count = int(
        (
            camera_events_df["risk_level"]
            .astype(str)
            .str.upper()
            == "MEDIUM"
        ).sum()
    )

    low_count = int(
        (
            camera_events_df["risk_level"]
            .astype(str)
            .str.upper()
            == "LOW"
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🚨 Total Incidents",
            len(camera_events_df)
        )

    with c2:
        st.metric(
            "🔴 High Risk",
            high_count
        )

    with c3:
        st.metric(
            "🟡 Medium Risk",
            medium_count
        )

    with c4:
        st.metric(
            "🟢 Low Risk",
            low_count
        )

    st.divider()

    # --------------------------------------------------------
    # INCIDENT HISTORY
    # --------------------------------------------------------

    st.subheader("📜 Incident History")

    display_df = camera_events_df.copy()

    display_df = display_df.rename(
        columns={
            "id": "ID",
            "timestamp": "Time",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "accuracy": "GPS Accuracy",
            "risk_level": "Risk Level",
            "risk_score": "Risk Score",
            "vehicles": "Vehicles",
            "pedestrians": "Pedestrians",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# ABOUT PAGE
# ============================================================

def about_page():

    st.header(
        "ℹ️ About RoadGuard AI"
    )

    st.markdown(
        """
        ## 🛡️ RoadGuard AI

        RoadGuard AI is an AI-assisted road safety
        and traffic intelligence platform.

        The system combines:

        - 🚦 Traffic information
        - ⚠️ Accident-risk prediction
        - 📷 Computer vision
        - 📍 Geographic information
        - 🗺️ Risk mapping
        - 📊 Analytics
        - 🚨 Safety scanning

        The architecture follows:

        **Data → Processing → Machine Learning →
        Prediction → Visualization → Safety Support**
        """
    )

    st.divider()

    st.subheader(
        "⚙️ System Architecture"
    )

    st.code(
        """
Road Data
    ↓
Preprocessing
    ↓
Machine Learning
    ├── Traffic Model
    └── Accident Model
    ↓
Computer Vision
    ↓
GPS / Location
    ↓
Risk Analysis
    ↓
RoadGuard AI Dashboard
        """,
        language="text",
    )

    st.warning(
        "RoadGuard AI is a decision-support prototype. "
        "Predictions and computer-vision outputs should "
        "not be treated as guaranteed real-world outcomes."
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    "## 🛡️ RoadGuard AI"
)

st.sidebar.caption(
    "Intelligent Road Safety Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Risk Prediction",
        "🚨 Safety Scan",
        "📷 Road Camera",
        "👁️ AI Vision",
        "📍 GPS Safety",
        "🚨 Alerts & Incidents",
        "Risk Map",
        "Analytics",
        "Model Performance",
        "About",
    ],
)


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "Dashboard":

    dashboard_page()

elif page == "Risk Prediction":

    prediction_page()

elif page == "🚨 Safety Scan":

    safety_scan_page()

elif page == "📷 Road Camera":

    camera_page()

elif page == "👁️ AI Vision":

    vision_page()

elif page == "📍 GPS Safety":

    gps_page()
    
elif page == "🚨 Alerts & Incidents":

    alerts_page()

elif page == "Risk Map":

    map_page()

elif page == "Analytics":

    analytics_page()

elif page == "Model Performance":

    performance_page()

else:

    about_page()


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "RoadGuard AI • Decision-support prototype"
)

st.sidebar.caption(
    f"Updated: {datetime.now().strftime('%d %b %Y')}"
)

