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
from typing import List

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
import json

# Pydantic models for structured data routing

class CropInfo(BaseModel):
    crop_type: str = Field(description="Type of crop, e.g., maize, tomatoes, wheat")
    growth_stage: str = Field(description="Growth stage: initial, development, mid, or late")
    area_sq_meters: float = Field(description="Area of the crop field in square meters")

class FarmerRequest(BaseModel):
    crop: CropInfo
    location: str = Field(description="Location of the farm for weather forecast")

class DailyForecast(BaseModel):
    day: int
    precipitation_mm: float
    et0_mm: float
    temp_max: float
    description: str

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
    farmer_guidance: str = Field(description="Friendly advice for the farmer based on weather")

# 1. Fetch Weather Data Node (Phase 0 Stub)
@node
def fetch_weather_data(ctx: Context, node_input: types.Content) -> WeatherData:
    """Mock node simulating weather forecast retrieval."""
    # Extract prompt text and parse JSON input manually
    prompt_text = ""
    if node_input and node_input.parts:
        prompt_text = node_input.parts[0].text or ""
    
    try:
        data = json.loads(prompt_text)
        req = FarmerRequest(**data)
    except Exception:
        # Fallback values if parsing fails (e.g. if plain text is sent)
        req = FarmerRequest(
            crop=CropInfo(crop_type="Maize", growth_stage="initial", area_sq_meters=10.0),
            location="DryVille"
        )
        
    forecast = []
    for day in range(1, 8):
        forecast.append(DailyForecast(
            day=day,
            precipitation_mm=0.0,
            et0_mm=4.0,
            temp_max=30.0,
            description="Mock sunny weather"
        ))
    ctx.state["crop"] = req.crop.model_dump()
    return WeatherData(forecast=forecast)

# 2. Crop Water Balance Calculation Node (Phase 0 Stub)
@node
def calculate_water_balance(ctx: Context, node_input: WeatherData) -> WaterBalanceResult:
    """Mock node simulating FAO-56 crop water balance calculation."""
    crop_dict = ctx.state.get("crop", {})
    area = crop_dict.get("area_sq_meters", 10.0)
    
    daily_balances = []
    for day_forecast in node_input.forecast:
        daily_balances.append(DailyWaterBalance(
            day=day_forecast.day,
            etc_mm=2.5,
            depletion_mm=10.0,
            irrigation_needed_mm=2.0,
            irrigation_needed_liters=2.0 * area
        ))
    
    return WaterBalanceResult(
        daily_balances=daily_balances,
        total_irrigation_liters=14.0 * area
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
    output_key="irrigation_schedule"
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
        water_status = f"💧 **{day.amount_liters:.1f} Liters**" if day.amount_liters > 0 else "🛑 No watering needed"
        schedule_text += f"- **Day {day.day_number}:** {water_status} | {day.recommendation}\n"
        
    schedule_text += f"\n#### Guidance & Tips:\n{node_input.farmer_guidance}"
    
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=schedule_text)]))
    yield Event(output=node_input)

# Create the Workflow Graph
workflow = Workflow(
    name="irrigation_optimizer_workflow",
    edges=[
        (START, fetch_weather_data),
        (fetch_weather_data, calculate_water_balance),
        (calculate_water_balance, scheduler_agent),
        (scheduler_agent, format_output)
    ],
    output_schema=IrrigationSchedule
)

# Wrap in App (the app name matches the directory 'app')
app = App(
    name="app",
    root_agent=workflow
)
