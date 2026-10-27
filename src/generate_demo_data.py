from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "traffic_accident_data.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main(n: int = 1800) -> None:
    rng = np.random.default_rng(42)

    start = pd.Timestamp("2025-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    hours = rng.integers(0, 24, n)
    minutes = rng.choice([0, 10, 20, 30, 40, 50], n)

    locations = np.array([
        "Patia Junction", "Jaydev Vihar", "Khandagiri Square", "Rasulgarh Square",
        "Nayapalli", "Acharya Vihar", "Baramunda", "Saheed Nagar"
    ])
    weather = rng.choice(["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"], n, p=[.45, .22, .17, .10, .06])
    road_condition = rng.choice(["Good", "Wet", "Poor"], n, p=[.55, .28, .17])
    road_type = rng.choice(["Urban", "Highway", "Residential", "Arterial"], n, p=[.45, .15, .12, .28])
    intersection_type = rng.choice(["Signalized", "Roundabout", "T-Junction", "Crossroad"], n)
    accident_history = rng.choice(["Low", "Medium", "High"], n, p=[.55, .30, .15])
    traffic_violations = rng.choice(["Low", "Medium", "High"], n, p=[.55, .30, .15])
    construction_zone = rng.choice(["No", "Yes"], n, p=[.86, .14])

    peak = ((hours >= 7) & (hours <= 10)) | ((hours >= 17) & (hours <= 21))
    traffic_volume = np.clip(rng.normal(430, 170, n) + peak * rng.normal(170, 55, n), 60, 1100).round().astype(int)
    vehicle_count = np.clip(traffic_volume * rng.uniform(.75, 1.35, n), 50, 1400).round().astype(int)
    average_speed = np.clip(rng.normal(49, 13, n) - peak * rng.normal(8, 3, n), 10, 95).round(1)
    queue_length = np.clip(rng.normal(12, 7, n) + peak * rng.normal(12, 4, n), 0, 60).round().astype(int)
    lane_occupancy = np.clip(rng.normal(.48, .15, n) + peak * rng.normal(.10, .04, n), .05, .98).round(3)
    temperature = np.clip(rng.normal(28, 4, n), 14, 42).round(1)
    rainfall = np.where(np.isin(weather, ["Rainy", "Stormy"]), rng.gamma(2.2, 3.5, n), rng.gamma(1.1, .6, n)).round(2)
    visibility = np.clip(rng.normal(9.5, 2.0, n), 1.5, 15)
    visibility[np.isin(weather, ["Rainy", "Stormy"])] *= rng.uniform(.45, .80, np.sum(np.isin(weather, ["Rainy", "Stormy"])))
    visibility[weather == "Foggy"] *= rng.uniform(.25, .60, np.sum(weather == "Foggy"))
    visibility = np.clip(visibility, .5, 15).round(2)
    road_width = np.clip(rng.normal(10.5, 3, n), 4, 20).round(1)
    number_of_lanes = rng.integers(1, 5, n)
    pedestrian_density = rng.choice(["Low", "Medium", "High"], n, p=[.50, .35, .15])

    lat = rng.normal(20.296, .035, n).round(6)
    lon = rng.normal(85.824, .045, n).round(6)

    risk_num = (
        0.8 * (traffic_volume / 1000)
        + 0.55 * (average_speed / 80)
        + 0.45 * (queue_length / 60)
        + 0.65 * lane_occupancy
        + 0.25 * (rainfall / 20)
        + 0.55 * (1 - visibility / 15)
        + np.where(road_condition == "Wet", .45, np.where(road_condition == "Poor", .70, 0))
        + np.where(accident_history == "High", .85, np.where(accident_history == "Medium", .35, 0))
        + np.where(traffic_violations == "High", .55, np.where(traffic_violations == "Medium", .20, 0))
        + np.where(construction_zone == "Yes", .25, 0)
        + np.where(pedestrian_density == "High", .30, np.where(pedestrian_density == "Medium", .12, 0))
        + np.where(weather == "Foggy", .45, np.where(weather == "Stormy", .35, np.where(weather == "Rainy", .18, 0)))
        + rng.normal(0, .18, n)
    )

    q1, q2 = np.quantile(risk_num, [.55, .82])
    accident_risk = np.select([risk_num < q1, risk_num < q2], ["Low", "Medium"], default="High")

    traffic_num = (
        traffic_volume / 250
        + queue_length / 25
        + 2.0 * lane_occupancy
        - average_speed / 70
        + peak * .9
        + rng.normal(0, .35, n)
    )
    tq1, tq2 = np.quantile(traffic_num, [.45, .78])
    traffic_level = np.select([traffic_num < tq1, traffic_num < tq2], ["Low", "Medium"], default="High")

    accident_count = np.clip((risk_num ** 2) * 1.7 + rng.normal(1, 1.5, n), 0, None).round().astype(int)
    accident_severity = np.select(
        [accident_count == 0, accident_count <= 4, accident_count <= 9],
        ["None", "Minor", "Moderate"],
        default="Severe",
    )

    df = pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "date": dates.date.astype(str),
        "time": [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)],
        "latitude": lat,
        "longitude": lon,
        "location": rng.choice(locations, n),
        "traffic_volume": traffic_volume,
        "traffic_density": np.select([traffic_volume < 350, traffic_volume < 650], ["Low", "Medium"], default="High"),
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
        "traffic_level": traffic_level,
        "accident_risk": accident_risk,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Demo dataset written to: {OUT}")
    print("WARNING: This synthetic dataset is for software testing only, not for claiming research accuracy.")


if __name__ == "__main__":
    main()
