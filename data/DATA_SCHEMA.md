# CSV schema

Required columns:

- record_id
- date
- time
- latitude
- longitude
- location
- traffic_volume
- traffic_density
- vehicle_count
- average_speed
- queue_length
- lane_occupancy
- weather
- temperature
- rainfall
- visibility
- road_condition
- road_type
- road_width
- number_of_lanes
- intersection_type
- construction_zone
- accident_history
- traffic_violations
- pedestrian_density
- accident_count
- accident_severity
- traffic_level
- accident_risk

Categorical targets must be Low/Medium/High, case-insensitive.

Example values:
- traffic_density: Low, Medium, High
- weather: Clear, Cloudy, Rainy, Foggy, Stormy
- road_condition: Good, Wet, Poor
- construction_zone: Yes, No
- accident_history: Low, Medium, High
- traffic_violations: Low, Medium, High
- accident_severity: None, Minor, Moderate, Severe
- traffic_level: Low, Medium, High
- accident_risk: Low, Medium, High
