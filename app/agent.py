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
import logging
import re
import requests
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

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

from google.adk.workflow import Workflow, node, START, Edge
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger("google_adk." + __name__)

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
    time_list = daily.get("time")

    if not isinstance(et0_list, list) or not isinstance(precip_list, list):
        raise ValueError("Malformed Open-Meteo response: daily fields must be lists")

    if len(et0_list) != len(precip_list) or len(et0_list) != days:
        raise ValueError(
            f"Malformed Open-Meteo response: daily list lengths mismatch "
            f"(expected {days}, got {len(et0_list)})"
        )

    if not time_list:
        today_val = date.today()
        time_list = [(today_val + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

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
                "date": str(time_list[idx]),
                "et0_mm": float(et0),
                "precipitation_mm": float(precip),
            }
        )

    # 6. Save to cache and return
    FORECAST_CACHE[cache_key] = (now, forecast_results)
    return forecast_results


# Dynamic Crop Data Loader
def load_supported_crops() -> List[str]:
    """Load supported crops list directly from the crop-profile skill references."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    crop_data_path = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            ".agents",
            "skills",
            "crop-profile",
            "references",
            "crop_data.json",
        )
    )
    try:
        with open(crop_data_path, "r") as f:
            data = json.load(f)
            return list(data.keys())
    except Exception as e:
        logger.warning(f"Failed to load supported crops from {crop_data_path}: {e}")
        return ["wheat", "maize", "cotton", "sugarcane", "tomato", "chickpea", "groundnut"]


# Deterministic Natural Language Parser (regex/substring checks, no hidden model calls)
def parse_profile_from_text(text: str) -> dict:
    extracted = {}

    # 1. Crop lookup (case-insensitive)
    supported_crops = load_supported_crops()
    for crop in supported_crops:
        if re.search(r'\b' + re.escape(crop) + r'\b', text, re.IGNORECASE):
            extracted["crop"] = crop
            break

    # 2. Field size regex (matches "5 ha", "2.5 hectares", "size is 10ha")
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ha|hectare)', text, re.IGNORECASE)
    if size_match:
        extracted["field_size_ha"] = float(size_match.group(1))

    # 3. Planting date regex (YYYY-MM-DD or natural language dates)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if date_match:
        try:
            datetime.strptime(date_match.group(1), "%Y-%m-%d")
            extracted["planting_date"] = date_match.group(1)
        except ValueError:
            pass
    else:
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        # Try natural date month-first (e.g. March 1, 2026 or March 1st, 2026)
        nat_match1 = re.search(
            r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b',
            text,
            re.IGNORECASE
        )
        if nat_match1:
            m_str = nat_match1.group(1).lower()
            d_str = nat_match1.group(2)
            y_str = nat_match1.group(3)
            m_num = month_map.get(m_str[:3])
            if m_num:
                extracted["planting_date"] = f"{int(y_str):04d}-{m_num:02d}-{int(d_str):02d}"
        else:
            # Try natural date day-first (e.g. 1 March 2026)
            nat_match2 = re.search(
                r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*,?\s*(\d{4})\b',
                text,
                re.IGNORECASE
            )
            if nat_match2:
                d_str = nat_match2.group(1)
                m_str = nat_match2.group(2).lower()
                y_str = nat_match2.group(3)
                m_num = month_map.get(m_str[:3])
                if m_num:
                    extracted["planting_date"] = f"{int(y_str):04d}-{m_num:02d}-{int(d_str):02d}"

    # 4. Lat/Lon regex (strictly requires explicit labels, allows optional quotes)
    lat_match = re.search(r'"?(?:lat(?:itude)?)"?\s*[:=]?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    lon_match = re.search(r'"?(?:lon(?:gitude)?)"?\s*[:=]?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if lat_match:
        extracted["latitude"] = float(lat_match.group(1))
    if lon_match:
        extracted["longitude"] = float(lon_match.group(1))

    # Fallback to coordinate pair match if labeled lat/lon are not found
    if "latitude" not in extracted or "longitude" not in extracted:
        coord_match = re.search(r'[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]', text)
        if coord_match:
            extracted["latitude"] = float(coord_match.group(1))
            extracted["longitude"] = float(coord_match.group(2))

    # 5. PII Consent / Opt-in detection
    if re.search(r'\b(?:save|persist|remember|store)\s+(?:my\s+)?profile\b', text, re.IGNORECASE) or \
       re.search(r'\bopt[- ]*in\b', text, re.IGNORECASE):
        extracted["profile_saved_opt_in"] = True
    elif re.search(r'\b(?:don\'?t\s+save|delete|remove|clear)\s+(?:my\s+)?profile\b', text, re.IGNORECASE) or \
         re.search(r'\bopt[- ]*out\b', text, re.IGNORECASE):
        extracted["profile_saved_opt_in"] = False

    return extracted


# Pydantic Schemas

class ProfileValidation(BaseModel):
    is_valid: bool = Field(description="True if all profile fields are present and valid, False otherwise")
    missing_fields: List[str] = Field(description="List of fields that are missing or invalid")
    reason: str = Field(description="Explanation of why the profile is invalid/complete")
    clarifying_question: str = Field(description="Friendly clarifying question to ask the farmer for the missing or invalid information. Empty if valid.")


class IrrigationRecommendation(BaseModel):
    crop_type: str = Field(description="The crop type")
    explanation: str = Field(description="A short, farmer-friendly explanation of the irrigation schedule, specifying which day(s) to irrigate, how much in liters, and the water-saved comparison.")


class IrrigationStateSchema(BaseModel):
    crop: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    field_size_ha: Optional[float] = None
    planting_date: Optional[str] = None
    profile_saved_opt_in: Optional[bool] = None
    security_blocked: Optional[bool] = None
    profile_validation: Optional[ProfileValidation] = None
    irrigation_recommendation: Optional[IrrigationRecommendation] = None
    current_user_message: Optional[str] = None


class DailyForecast(BaseModel):
    day: int
    date: Optional[str] = None
    precipitation_mm: float
    et0_mm: float


class WeatherData(BaseModel):
    forecast: List[DailyForecast]


# Deterministic validation backstop independent of the LLM
def validate_profile_deterministically(profile: dict) -> Tuple[bool, List[str], str, str]:
    """Deterministically validates profile fields using explicit checks and Pydantic rules."""
    missing_fields = []
    invalid_fields = []

    # 1. Completeness check
    for field in ["crop", "latitude", "longitude", "field_size_ha", "planting_date"]:
        if profile.get(field) is None:
            missing_fields.append(field)

    # 2. Sane range and type checks on provided fields
    supported = load_supported_crops()

    crop = profile.get("crop")
    if crop is not None:
        if str(crop).lower() not in [c.lower() for c in supported]:
            invalid_fields.append("crop")

    lat = profile.get("latitude")
    if lat is not None:
        try:
            val = float(lat)
            if not (-90.0 <= val <= 90.0):
                invalid_fields.append("latitude")
        except ValueError:
            invalid_fields.append("latitude")

    lon = profile.get("longitude")
    if lon is not None:
        try:
            val = float(lon)
            if not (-180.0 <= val <= 180.0):
                invalid_fields.append("longitude")
        except ValueError:
            invalid_fields.append("longitude")

    size = profile.get("field_size_ha")
    if size is not None:
        try:
            val = float(size)
            if val <= 0.0:
                invalid_fields.append("field_size_ha")
        except ValueError:
            invalid_fields.append("field_size_ha")

    pdate = profile.get("planting_date")
    if pdate is not None:
        try:
            if isinstance(pdate, date):
                pass
            else:
                datetime.strptime(str(pdate), "%Y-%m-%d")
        except ValueError:
            invalid_fields.append("planting_date")

    all_failures = missing_fields + invalid_fields
    if all_failures:
        reasons = []
        clarifications = []

        if "crop" in all_failures:
            if crop:
                reasons.append(f"{crop} is not a supported crop.")
                clarifications.append(f"choose one of the supported crops: {', '.join(supported)}")
            else:
                reasons.append("Crop is missing.")
                clarifications.append(f"provide a supported crop ({', '.join(supported)})")
        if "latitude" in all_failures or "longitude" in all_failures:
            reasons.append("Coordinates are missing or out of valid ranges (latitude: [-90, 90], longitude: [-180, 180]).")
            clarifications.append("provide valid coordinates (latitude and longitude)")
        if "field_size_ha" in all_failures:
            reasons.append("Field size must be a positive number of hectares.")
            clarifications.append("provide a positive field size in hectares")
        if "planting_date" in all_failures:
            reasons.append("Planting date is missing or not in YYYY-MM-DD format.")
            clarifications.append("provide a valid planting date in YYYY-MM-DD format")

        reason_str = " ".join(reasons)
        clarifying_question = "Please " + ", and ".join(clarifications) + "."

        return False, all_failures, reason_str, clarifying_question

    return True, [], "", ""


# Global to communicate crop type to monkeypatched LLM mock generator
LAST_RUN_CROP = "wheat"


# Workflow Nodes

# 1. Save Profile Node
@node
def save_profile(ctx: Context, node_input: types.Content) -> dict:
    """Validates and stores the farmer's profile in session state."""
    global LAST_RUN_CROP
    text = ""
    if node_input and hasattr(node_input, "parts") and node_input.parts:
        text = (node_input.parts[0].text or "").strip()
    elif isinstance(node_input, str):
        text = node_input

    # Reset security blocked flag on new turn
    ctx.state["security_blocked"] = False
    ctx.state["current_user_message"] = text

    # Reconstruct state from history if state is empty (e.g. during evaluation replay)
    if ctx.session and ctx.session.events:
        for ev in ctx.session.events:
            # Check if event is user message
            author = getattr(ev, "author", None)
            if author is None and isinstance(ev, dict):
                author = ev.get("author")
            if author == "user":
                content = getattr(ev, "content", None)
                if content is None and isinstance(ev, dict):
                    content = ev.get("content")
                parts = getattr(content, "parts", None)
                if parts is None and isinstance(content, dict):
                    parts = content.get("parts")

                parts_list = []
                if parts:
                    for part in parts:
                        p_text = getattr(part, "text", None)
                        if p_text is None and isinstance(part, dict):
                            p_text = part.get("text")
                        if p_text:
                            parts_list.append(p_text)

                text_hist = "".join(parts_list)
                if text_hist:
                    extracted_hist = parse_profile_from_text(text_hist)
                    for k, v in extracted_hist.items():
                        if v is not None:
                            ctx.state[k] = v

    profile = {
        "crop": ctx.state.get("crop"),
        "latitude": ctx.state.get("latitude"),
        "longitude": ctx.state.get("longitude"),
        "field_size_ha": ctx.state.get("field_size_ha"),
        "planting_date": ctx.state.get("planting_date"),
        "profile_saved_opt_in": ctx.state.get("profile_saved_opt_in"),
    }

    # Try parsing as JSON first (useful for API/tests)
    parsed_json = False
    if text.strip().startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Check for nested crop object
                if "crop" in data and isinstance(data["crop"], dict):
                    crop_info = data["crop"]
                    if "crop_type" in crop_info:
                        profile["crop"] = crop_info["crop_type"]
                    if "planting_date" in crop_info:
                        profile["planting_date"] = str(crop_info["planting_date"])
                    if "field_size_ha" in crop_info:
                        profile["field_size_ha"] = float(crop_info["field_size_ha"])

                # Direct fields
                for k in ["crop", "latitude", "longitude", "field_size_ha", "planting_date", "profile_saved_opt_in"]:
                    if k in data and data[k] is not None:
                        if k == "crop" and isinstance(data[k], dict):
                            continue
                        if k == "crop":
                            profile["crop"] = data[k]
                        elif k in ("latitude", "longitude", "field_size_ha"):
                            profile[k] = float(data[k])
                        elif k == "profile_saved_opt_in":
                            profile[k] = bool(data[k])
                        else:
                            profile[k] = str(data[k])
                parsed_json = True
        except Exception:
            pass

    if not parsed_json and text.strip():
        extracted = parse_profile_from_text(text)
        for k, v in extracted.items():
            profile[k] = v

    # Store valid fields in state
    for k, v in profile.items():
        if v is not None:
            ctx.state[k] = v

    if ctx.state.get("crop"):
        LAST_RUN_CROP = ctx.state["crop"]

    return profile


# 2. Validate Profile LlmAgent Node
supported_crops = load_supported_crops()
supported_crops_str = ", ".join(supported_crops)
validate_profile = LlmAgent(
    name="validate_profile",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are an agronomy system validator. Your task is to judge whether the farmer's profile "
        "is complete and valid. The profile must contain:\n"
        f"1. crop: Must be one of the supported crops: {supported_crops_str}.\n"
        "2. latitude: Must be a number between -90.0 and 90.0.\n"
        "3. longitude: Must be a number between -180.0 and 180.0.\n"
        "4. field_size_ha: Must be a positive number (greater than 0).\n"
        "5. planting_date: Must be a valid date in YYYY-MM-DD format.\n\n"
        "Input will be the stored profile. If any field is missing or invalid, set is_valid to False, "
        "list the missing_fields, explain the reason, and write a friendly clarifying question asking the "
        "farmer to provide the missing or correct information. If all fields are valid, set is_valid to True "
        "and clarifying_question to empty."
    ),
    output_schema=ProfileValidation,
    output_key="profile_validation",
)


# 3. Route Validation Node
@node
def route_validation(ctx: Context, node_input: Any) -> Any:
    """Sets the workflow route based on the validation result, with a deterministic backstop."""
    logger.info(f"ROUTE_VALIDATION: node_input={node_input}")

    # 1. Run deterministic validation backstop first
    profile = {
        "crop": ctx.state.get("crop"),
        "latitude": ctx.state.get("latitude"),
        "longitude": ctx.state.get("longitude"),
        "field_size_ha": ctx.state.get("field_size_ha"),
        "planting_date": ctx.state.get("planting_date"),
    }
    det_valid, det_failures, det_reason, det_question = validate_profile_deterministically(profile)

    if not det_valid:
        logger.warning(f"ROUTE_VALIDATION: Deterministic backstop failed: {det_failures} - {det_reason}")
        val = ProfileValidation(
            is_valid=False,
            missing_fields=det_failures,
            reason=det_reason,
            clarifying_question=det_question
        )
        ctx.state["profile_validation"] = val.model_dump()
        ctx.route = False
        return val

    # 2. Fallback to LLM validation output
    val = node_input
    if val is None:
        val = ctx.state.get("profile_validation")
    if val is None:
        logger.warning("ROUTE_VALIDATION: profile_validation is None!")
        ctx.route = False
        return None

    if hasattr(val, "is_valid"):
        ctx.route = val.is_valid
    elif isinstance(val, dict):
        ctx.route = val.get("is_valid", False)
    else:
        ctx.route = False
    return val


# 4. Clarify Profile Node
@node
def clarify_profile(ctx: Context, node_input: Any) -> None:
    """Yields the clarifying question to the user."""
    logger.info(f"CLARIFY_PROFILE: node_input={node_input}")
    val = node_input
    if val is None:
        val = ctx.state.get("profile_validation")

    question = ""
    if val and hasattr(val, "clarifying_question"):
        question = val.clarifying_question
    elif isinstance(val, dict):
        question = val.get("clarifying_question", "")

    if not question:
        question = "Please provide your crop type, location coordinates (latitude and longitude), field size (in hectares), and planting date to generate your irrigation schedule."

    yield Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=question)]
        )
    )


# 5. Fetch Weather Node
@node
def fetch_weather(ctx: Context, node_input: Any) -> WeatherData:
    """Calls get_forecast for the farmer's location."""
    lat = ctx.state["latitude"]
    lon = ctx.state["longitude"]
    forecast_data = get_forecast(lat, lon)
    forecast_list = [DailyForecast(**item) for item in forecast_data]
    return WeatherData(forecast=forecast_list)


# 6. Compute Schedule Node
@node
def compute_schedule(ctx: Context, node_input: WeatherData) -> dict:
    """Calls the irrigation-calculator skill with the weather forecast and profile."""
    crop = ctx.state["crop"].lower()

    # Calculate days_after_planting
    planting_date_val = ctx.state["planting_date"]
    if isinstance(planting_date_val, date):
        planting_date = planting_date_val
    else:
        planting_date = datetime.strptime(str(planting_date_val), "%Y-%m-%d").date()

    today = date.today()
    days_after_planting = (today - planting_date).days
    if days_after_planting < 0:
        days_after_planting = 0

    field_size_ha = float(ctx.state["field_size_ha"])

    # Prepare forecast payload for calc_schedule.py (requires 'date', 'et0_mm', 'precip_mm')
    forecast_payload = []
    for day in node_input.forecast:
        forecast_payload.append({
            "date": day.date,
            "et0_mm": day.et0_mm,
            "precip_mm": day.precipitation_mm
        })

    import subprocess
    import sys

    script_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            ".agents",
            "skills",
            "irrigation-calculator",
            "scripts",
            "calc_schedule.py"
        )
    )

    payload = {
        "crop": crop,
        "days_after_planting": days_after_planting,
        "field_size_ha": field_size_ha,
        "forecast": forecast_payload
    }

    result = subprocess.run(
        [sys.executable, script_path],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True
    )

    schedule_output = json.loads(result.stdout)

    # Output Guardrail: Enforce MAX_DAILY_MM = 60.0 and log warning
    MAX_DAILY_MM = 60.0
    for day in schedule_output.get("schedule", []):
        net_mm = day.get("net_irrigation_mm", 0.0)
        etc_mm = day.get("etc_mm", 0.0)
        eff_rain = day.get("effective_rain_mm", 0.0)
        calculated_uncapped_mm = etc_mm - eff_rain

        if calculated_uncapped_mm > MAX_DAILY_MM or net_mm > MAX_DAILY_MM:
            logger.warning(
                f"Output Guardrail Triggered: Daily irrigation depth for date {day.get('date')} "
                f"exceeded MAX_DAILY_MM ({MAX_DAILY_MM} mm). Uncapped: {calculated_uncapped_mm:.2f} mm. Capping."
            )
            day["net_irrigation_mm"] = MAX_DAILY_MM
            day["liters_needed"] = round(MAX_DAILY_MM * field_size_ha * 10000.0, 1)
    schedule_output["crop"] = crop
    return schedule_output


# 6.5. Security Screening Node
@node
def security_screen(ctx: Context, node_input: Any) -> Any:
    """Screens the current turn's latest user message for prompt-injection patterns."""
    user_text = ctx.state.get("current_user_message") or ""
    if not user_text and ctx.session and ctx.session.events:
        # Scan backward to find strictly the latest user message from the current turn
        for ev in reversed(ctx.session.events):
            author = getattr(ev, "author", None)
            if author is None and isinstance(ev, dict):
                author = ev.get("author")
            if author == "user":
                content = getattr(ev, "content", None)
                if content is None and isinstance(ev, dict):
                    content = ev.get("content")
                parts = getattr(content, "parts", None)
                if parts is None and isinstance(content, dict):
                    parts = content.get("parts")
                parts_list = []
                if parts:
                    for part in parts:
                        p_text = getattr(part, "text", None)
                        if p_text is None and isinstance(part, dict):
                            p_text = part.get("text")
                        if p_text:
                            parts_list.append(p_text)
                user_text = "".join(parts_list)
                break  # Scope strictly to current turn's input

    injection_patterns = [
        r"(?i)ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"(?i)system\s+override",
        r"(?i)you\s+are\s+now\s+a\s+",
        r"(?i)you\s+are\s+now\s+dan\b",
        r"(?i)act\s+as\s+a\s+",
        r"(?i)act\s+as\s+dan\b",
        r"(?i)bypass\s+security",
        r"(?i)prompt\s+injection",
        r"(?i)ignore\s+(?:the\s+)?above\s+(?:instructions|prompt|message)",
        r"(?i)new\s+role",
        r"(?i)dan\s+mode",
    ]

    is_injection = False
    for pattern in injection_patterns:
        if re.search(pattern, user_text):
            is_injection = True
            break

    if is_injection:
        logger.warning(f"Security Warning: Prompt injection detected on current turn: {user_text!r}")
        ctx.route = False  # Route to canned_injection_response
        ctx.state["security_blocked"] = True
        return {"injection_detected": True, "schedule_data": node_input}
    else:
        ctx.route = True   # Route to recommend LLM Agent
        ctx.state["security_blocked"] = False
        return node_input


# 6.6. Canned Injection Response Node
@node
def canned_injection_response(ctx: Context, node_input: Any) -> Any:
    """Sets a safe canned response when prompt-injection is detected."""
    response = IrrigationRecommendation(
        crop_type=ctx.state.get("crop", "unknown"),
        explanation="I am sorry, but I cannot fulfill this request. I am only able to provide irrigation recommendations and answer related agricultural questions."
    )
    ctx.state["irrigation_recommendation"] = response.model_dump()
    return response


def ground_crop_type(callback_context, llm_response):
    if llm_response and llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                try:
                    data = json.loads(part.text)
                    data["crop_type"] = callback_context.state.get("crop", "unknown")
                    part.text = json.dumps(data)
                except (json.JSONDecodeError, TypeError):
                    pass
    return None

recommend = LlmAgent(
    name="recommend",
    model="gemini-3.1-flash-lite",
    instruction=("You are an agronomy expert helper. The input data includes a 'crop' field stating the "
    "farmer's exact crop type — you must copy that value verbatim into your crop_type output. "
    "Do not infer, guess, or substitute a different crop under any circumstances. "
    "Your task is to take the numeric irrigation schedule and water savings comparison and turn "
    "them into a short, friendly, and practical explanation for the farmer. Be concise and clear "
    "about which day(s) to irrigate, how many liters to apply, and how much water is saved "
    "compared to a standard daily watering baseline."),
    output_schema=IrrigationRecommendation,
    output_key="irrigation_recommendation",
    after_model_callback=ground_crop_type,
)


# 8. Format Recommendation Node
@node
def format_recommendation(ctx: Context, node_input: Any) -> Any:
    """Yields the friendly explanation to the user and manages PII consent persistence."""
    logger.info(f"FORMAT_RECOMMENDATION: node_input={node_input}")
    val = node_input
    if val is None:
        val = ctx.state.get("irrigation_recommendation")
    real_crop = ctx.state.get("crop", "unknown")
    if val is not None:
        if hasattr(val, "crop_type"):
            val.crop_type = real_crop
        elif isinstance(val, dict):
            val["crop_type"] = real_crop
    explanation = ""
    if val and hasattr(val, "explanation"):
        explanation = val.explanation
    elif isinstance(val, dict):
        explanation = val.get("explanation", "")

    if not explanation:
        explanation = "Could not generate an irrigation recommendation. Please check your profile information."

    # Append advisory disclaimer ONLY on genuine recommendations
    if not ctx.state.get("security_blocked", False):
        disclaimer = "Advisory: This recommendation supplements, does not replace, local agricultural extension advice."
        explanation = f"{explanation}\n\n{disclaimer}"

    # PII consent saving pattern
    user_id = ctx.session.user_id if (ctx.session and ctx.session.user_id) else "default_user"
    opt_in = ctx.state.get("profile_saved_opt_in", False)
    profiles_file = "saved_profiles.json"

    if opt_in:
        profile_data = {
            "crop": ctx.state.get("crop"),
            "latitude": ctx.state.get("latitude"),
            "longitude": ctx.state.get("longitude"),
            "field_size_ha": ctx.state.get("field_size_ha"),
            "planting_date": ctx.state.get("planting_date"),
        }
        try:
            if os.path.exists(profiles_file):
                with open(profiles_file, "r") as f:
                    profiles = json.load(f)
            else:
                profiles = {}
            profiles[user_id] = profile_data
            with open(profiles_file, "w") as f:
                json.dump(profiles, f, indent=2)
            logger.info(f"PII persisted for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to persist PII profile: {e}")
    else:
        # Consent not given: clean up profile from disk
        if os.path.exists(profiles_file):
            try:
                with open(profiles_file, "r") as f:
                    profiles = json.load(f)
                if user_id in profiles:
                    del profiles[user_id]
                    with open(profiles_file, "w") as f:
                        json.dump(profiles, f, indent=2)
                    logger.info(f"PII cleared for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to clean up PII profile: {e}")

    yield Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=explanation)]
        )
    )
    yield Event(output=val)


# Helpers for Chaining Nodes Explicitly
def chain_edges(*nodes: Any) -> List[Edge]:
    """Helper to chain a sequence of nodes unconditionally."""
    return [Edge(from_node=a, to_node=b) for a, b in zip(nodes, nodes[1:])]


# Build Edge List Explicitly
edges = (
    chain_edges(START, save_profile, security_screen)
    + [
        Edge(from_node=security_screen, to_node=validate_profile, route=True),
        Edge(from_node=security_screen, to_node=canned_injection_response, route=False),
    ]
    + chain_edges(validate_profile, route_validation)
    + [
        Edge(from_node=route_validation, to_node=fetch_weather, route=True),
        Edge(from_node=route_validation, to_node=clarify_profile, route=False),
    ]
    + chain_edges(fetch_weather, compute_schedule, recommend, format_recommendation)
    + [
        Edge(from_node=canned_injection_response, to_node=format_recommendation),
    ]
)


# Recompile Workflow Graph
workflow = Workflow(
    name="irrigation_optimizer_workflow",
    edges=edges,
    state_schema=IrrigationStateSchema,
    output_schema=IrrigationRecommendation,
)

# Wrap in App
app = App(name="app", root_agent=workflow)
root_agent = workflow


# Monkeypatch Gemini LLM call if E2E_MOCK environment variable is set
if os.environ.get("E2E_MOCK") == "TRUE":
    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_response import LlmResponse

    original_generate = Gemini.generate_content_async

    async def mocked_generate(self, llm_request, stream=False):
        sys_inst = llm_request.config.system_instruction or ""
        prompt_text = ""
        for content in llm_request.contents:
            for part in content.parts:
                if part.text:
                    prompt_text += part.text + "\n"

        if "agronomy system validator" in sys_inst.lower():
            import json

            user_text = ""
            for content in reversed(llm_request.contents):
                if content.role == "user":
                    for part in content.parts:
                        if part.text:
                            user_text += part.text + "\n"
                    break

            try:
                profile = json.loads(user_text.strip())
            except Exception:
                profile = {}

            supported = load_supported_crops()

            is_valid = True
            missing_fields = []
            clarifying_question = ""

            crop = profile.get("crop")
            lat = profile.get("latitude")
            lon = profile.get("longitude")
            size = profile.get("field_size_ha")
            pdate = profile.get("planting_date")

            if not crop:
                missing_fields.append("crop")
            elif crop.lower() not in [c.lower() for c in supported]:
                missing_fields.append("crop")

            if lat is None or not (-90.0 <= lat <= 90.0):
                missing_fields.append("latitude")
            if lon is None or not (-180.0 <= lon <= 180.0):
                missing_fields.append("longitude")
            if size is None or size <= 0:
                missing_fields.append("field_size_ha")
            if not pdate:
                missing_fields.append("planting_date")
            else:
                try:
                    from datetime import datetime
                    datetime.strptime(str(pdate), "%Y-%m-%d")
                except ValueError:
                    missing_fields.append("planting_date")

            if missing_fields:
                is_valid = False
                if crop and crop.lower() not in [c.lower() for c in supported]:
                    clarifying_question = f"{crop} is not supported. Please choose one of the supported crops: {', '.join(supported)}."
                else:
                    clarifying_question = f"Please provide the missing or incorrect fields: {', '.join(missing_fields)}."
            else:
                is_valid = True
                clarifying_question = ""

            validation_result = {
                "is_valid": is_valid,
                "missing_fields": missing_fields,
                "reason": "Profile validation check",
                "clarifying_question": clarifying_question
            }

            from google.genai import types
            resp_obj = types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            parts=[types.Part.from_text(text=json.dumps(validation_result))]
                        )
                    )
                ]
            )
            llm_response = LlmResponse.create(resp_obj)
            yield llm_response

        elif "agronomy expert helper" in sys_inst.lower():
            import json
            import re

            # Extract the crop type
            crop_type = LAST_RUN_CROP
            supported = load_supported_crops()
            crop_match = re.search(r'"crop"\s*:\s*"([a-zA-Z]+)"', prompt_text)
            if not crop_match:
                crop_match = re.search(r'crop\s*:\s*([a-zA-Z]+)', prompt_text, re.IGNORECASE)
            if crop_match:
                crop_type = crop_match.group(1).lower()
            else:
                for crop in supported:
                    if re.search(r'\b' + re.escape(crop) + r'\b', prompt_text, re.IGNORECASE):
                        crop_type = crop
                        break

            # Try to parse schedule JSON to detect if irrigation is needed
            schedule_data = {}
            for block in re.findall(r'(\{.*?\})', prompt_text, re.DOTALL):
                try:
                    parsed = json.loads(block)
                    if "schedule" in parsed:
                        schedule_data = parsed
                        break
                except Exception:
                    pass

            # Parse savings
            saved_liters = 12000.0
            pct_saved = 44.4
            saved_liters_match = re.search(r'"saved_liters":\s*(\d+(?:\.\d+)?)', prompt_text)
            if not saved_liters_match:
                saved_liters_match = re.search(r'saved_liters[^\d]*(\d+(?:\.\d+)?)', prompt_text, re.IGNORECASE)
            if saved_liters_match:
                saved_liters = float(saved_liters_match.group(1))

            pct_saved_match = re.search(r'"pct_saved":\s*(\d+(?:\.\d+)?)', prompt_text)
            if not pct_saved_match:
                pct_saved_match = re.search(r'pct_saved[^\d]*(\d+(?:\.\d+)?)', prompt_text, re.IGNORECASE)
            if pct_saved_match:
                pct_saved = float(pct_saved_match.group(1))

            irrigation_days = []
            if schedule_data and "schedule" in schedule_data:
                for day_info in schedule_data["schedule"]:
                    liters = day_info.get("liters_needed", 0.0)
                    day_num = day_info.get("day")
                    if liters > 0.0:
                        irrigation_days.append((day_num, liters))

            if not irrigation_days:
                explanation = f"Based on the 7-day forecast, no irrigation is needed for your {crop_type} field due to sufficient rainfall. You saved {saved_liters:,.0f} Liters ({pct_saved:.1f}%) of water."
            else:
                days_str = ", ".join([f"Day {d} ({liters:,.0f} Liters)" for d, liters in irrigation_days])
                explanation = f"Based on the 7-day forecast, you should irrigate your {crop_type} field on: {days_str}. This scheduling saves {saved_liters:,.0f} Liters ({pct_saved:.1f}%) of water compared to daily watering."

            recommendation_result = {
                "crop_type": crop_type,
                "explanation": explanation
            }
            from google.genai import types
            resp_obj = types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            parts=[types.Part.from_text(text=json.dumps(recommendation_result))]
                        )
                    )
                ]
            )
            llm_response = LlmResponse.create(resp_obj)
            yield llm_response
        else:
            async for r in original_generate(self, llm_request, stream):
                yield r

    Gemini.generate_content_async = mocked_generate
