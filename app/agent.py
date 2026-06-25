# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
from typing import List, Dict, Tuple

# Load .env file manually to support local api keys without external dependencies
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# Set non-enterprise local authentication variables
os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "FALSE"

from google.adk.workflow import Workflow, node, START
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.genai import types
from pydantic import BaseModel, Field
import requests

# In-memory cache for forecast requests
# Key: (lat, lon, days), Value: (timestamp, validated_forecast_list)
FORECAST_CACHE: Dict[Tuple[float, float, int], Tuple[float, List[dict]]] = {}
CACHE_TTL = 3600  # 1 hour in seconds


def get_forecast(lat: float, lon: float, days: int = 7) -> List[dict]:
    """Fetch 7-day forecast daily et0 and precipitation from Open-Meteo.

    Validates coordinates and response format. Uses an in-memory cache.
    """
    # 1. Coordinate range validation
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude {lat} is out of bounds [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude {lon} is out of bounds [-180, 180]")

    # 2. Check in-memory cache
    cache_key = (lat, lon, days)
    now = time.time()
    if cache_key in FORECAST_CACHE:
        timestamp, cached_data = FORECAST_CACHE[cache_key]
        if now - timestamp < CACHE_TTL:
            return cached_data

    # 3. Call Open-Meteo API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "et0_fao_evapotranspiration,precipitation_sum",
        "timezone": "auto",
        "forecast_days": days,
    }
    response = requests.get(url, params=params, timeout=10)
    if not response.ok:
        raise ValueError(
            f"Open-Meteo API returned error {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
    except Exception as e:
        raise ValueError(f"Failed to parse Open-Meteo response as JSON: {e}")

    # 4. Validate response shape and daily fields
    if "daily" not in data or not isinstance(data["daily"], dict):
        raise ValueError("Malformed Open-Meteo response: 'daily' section is missing")

    daily = data["daily"]
    if "et0_fao_evapotranspiration" not in daily or "precipitation_sum" not in daily:
        raise ValueError(
            "Malformed Open-Meteo response: daily keys 'et0_fao_evapotranspiration' "
            "or 'precipitation_sum' are missing"
        )

    et0_list = daily["et0_fao_evapotranspiration"]
    precip_list = daily["precipitation_sum"]

    if not isinstance(et0_list, list) or not isinstance(precip_list, list):
        raise ValueError("Malformed Open-Meteo response: daily fields must be lists")

    if len(et0_list) != len(precip_list) or len(et0_list) != days:
        raise ValueError(
            f"Malformed Open-Meteo response: daily list lengths mismatch "
            f"(expected {days}, got {len(et0_list)})"
        )

    # 5. Type and numeric validation for every item
    forecast_results = []
    for idx in range(days):
        et0 = et0_list[idx]
        precip = precip_list[idx]

        # Ensure values are strictly numeric and not boolean (bool is a subclass of int)
        if (
            not isinstance(et0, (int, float))
            or isinstance(et0, bool)
            or not isinstance(precip, (int, float))
            or isinstance(precip, bool)
        ):
            raise ValueError(
                f"Malformed Open-Meteo response: non-numeric daily value at index {idx} "
                f"(et0={et0}, precip={precip})"
            )

        forecast_results.append(
            {
                "day": idx + 1,
                "et0_mm": float(et0),
                "precipitation_mm": float(precip),
            }
        )

    # 6. Save to cache and return
    FORECAST_CACHE[cache_key] = (now, forecast_results)
    return forecast_results


# Pydantic models for structured data routing


class CropInfo(BaseModel):
    crop_type: str = Field(description="Type of crop, e.g., maize, tomatoes, wheat")
    growth_stage: str = Field(
        description="Growth stage: initial, development, mid, or late"
    )
    area_sq_meters: float = Field(description="Area of the crop field in square meters")


class FarmerRequest(BaseModel):
    crop: CropInfo
    latitude: float = Field(description="Latitude of the farm", ge=-90.0, le=90.0)
    longitude: float = Field(description="Longitude of the farm", ge=-180.0, le=180.0)


class DailyForecast(BaseModel):
    day: int
    precipitation_mm: float
    et0_mm: float


class WeatherData(BaseModel):
    forecast: List[DailyForecast]


class DailyWaterBalance(BaseModel):
    day: int
    etc_mm: float
    depletion_mm: float
    irrigation_needed_mm: float
    irrigation_needed_liters: float


class WaterBalanceResult(BaseModel):
    daily_balances: List[DailyWaterBalance]
    total_irrigation_liters: float


class IrrigationDay(BaseModel):
    day_number: int = Field(description="Day number (1 to 7)")
    amount_liters: float = Field(description="Amount of water to apply in liters")
    recommendation: str = Field(description="Timing or watering suggestion")


class IrrigationSchedule(BaseModel):
    crop_type: str
    schedule: List[IrrigationDay]
    total_water_liters: float
    farmer_guidance: str = Field(
        description="Friendly advice for the farmer based on weather"
    )


# 1. Fetch Weather Data Node
@node
def fetch_weather_data(ctx: Context, node_input: types.Content) -> WeatherData:
    """Node that fetches the daily forecast from Open-Meteo."""
    prompt_text = ""
    if node_input and node_input.parts:
        prompt_text = (node_input.parts[0].text or "").strip()

    if not prompt_text:
        # Fallback values only if payload is genuinely missing
        req = FarmerRequest(
            crop=CropInfo(
                crop_type="Tomatoes", growth_stage="development", area_sq_meters=10.0
            ),
            latitude=-1.2921,
            longitude=36.8219,
        )
    else:
        try:
            data = json.loads(prompt_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}") from e

        # This will raise Pydantic ValidationError if out-of-bounds/invalid fields are passed
        req = FarmerRequest(**data)

    forecast_data = get_forecast(req.latitude, req.longitude)
    forecast_list = []
    for item in forecast_data:
        forecast_list.append(DailyForecast(**item))

    ctx.state["crop"] = req.crop.model_dump()
    return WeatherData(forecast=forecast_list)


# 2. Crop Water Balance Calculation Node (Phase 0 Stub)
@node
def calculate_water_balance(
    ctx: Context, node_input: WeatherData
) -> WaterBalanceResult:
    """Mock node simulating FAO-56 crop water balance calculation."""
    crop_dict = ctx.state.get("crop", {})
    area = crop_dict.get("area_sq_meters", 10.0)

    daily_balances = []
    for day_forecast in node_input.forecast:
        daily_balances.append(
            DailyWaterBalance(
                day=day_forecast.day,
                etc_mm=2.5,
                depletion_mm=10.0,
                irrigation_needed_mm=2.0,
                irrigation_needed_liters=2.0 * area,
            )
        )

    return WaterBalanceResult(
        daily_balances=daily_balances, total_irrigation_liters=14.0 * area
    )


# 3. LLM Scheduler Agent Node (AI Studio Gemini Key)
scheduler_agent = LlmAgent(
    name="scheduler_agent",
    model="gemini-flash-latest",
    instruction=(
        "You are an expert agronomy assistant helping smallholder farmers optimize their irrigation. "
        "Your task is to take a weather forecast and a scientific FAO-56 crop water balance calculation "
        "and produce a simple, practical, 7-day irrigation schedule. "
        "Provide daily recommendations on whether to water and how much (in liters) for their crop, "
        "and give a friendly, simple tip for the farmer based on the weather conditions."
    ),
    output_schema=IrrigationSchedule,
    output_key="irrigation_schedule",
)


# 4. Format Output Node for final Web UI/output rendering
@node
def format_output(ctx: Context, node_input: IrrigationSchedule) -> IrrigationSchedule:
    """Node that formats the structured schedule output into user-friendly Markdown content."""
    schedule_text = (
        f"### 🌾 7-Day Irrigation Schedule for {node_input.crop_type}\n\n"
        f"**Total Water Required:** {node_input.total_water_liters:.1f} Liters\n\n"
        f"#### Daily Schedule:\n"
    )
    for day in node_input.schedule:
        water_status = (
            f"💧 **{day.amount_liters:.1f} Liters**"
            if day.amount_liters > 0
            else "🛑 No watering needed"
        )
        schedule_text += (
            f"- **Day {day.day_number}:** {water_status} | {day.recommendation}\n"
        )

    schedule_text += f"\n#### Guidance & Tips:\n{node_input.farmer_guidance}"

    yield Event(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=schedule_text)]
        )
    )
    yield Event(output=node_input)


# Create the Workflow Graph
workflow = Workflow(
    name="irrigation_optimizer_workflow",
    edges=[
        (START, fetch_weather_data),
        (fetch_weather_data, calculate_water_balance),
        (calculate_water_balance, scheduler_agent),
        (scheduler_agent, format_output),
    ],
    output_schema=IrrigationSchedule,
)

# Wrap in App (the app name matches the directory 'app')
app = App(name="app", root_agent=workflow)
root_agent = workflow
