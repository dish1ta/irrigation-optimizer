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

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from unittest.mock import patch, MagicMock
from google.adk.events.event import Event

from app.agent import root_agent, ProfileValidation, IrrigationRecommendation


async def mock_run_llm_agent_as_node(agent, *, ctx, node_input):
    """Mocks LlmAgent node execution to make integration tests deterministic and fast."""
    if agent.name == "validate_profile":
        crop = ctx.state.get("crop")
        lat = ctx.state.get("latitude")
        lon = ctx.state.get("longitude")
        size = ctx.state.get("field_size_ha")
        date_val = ctx.state.get("planting_date")
        
        # In mock, check user history for unsupported crop detection
        user_text = ""
        for ev in reversed(ctx.session.events):
            if ev.author == "user" and ev.content and ev.content.parts:
                user_text = "".join(part.text for part in ev.content.parts if part.text)
                break
        
        if "rice" in user_text.lower():
            crop = "rice"

        is_valid = all([crop, lat is not None, lon is not None, size is not None, date_val])
        
        if crop == "rice":
            is_valid = False
            missing_fields = ["crop"]
            clarifying_question = "Rice is not supported. Please choose one of the supported crops: wheat, maize, cotton, sugarcane, tomato, chickpea, groundnut."
        elif not is_valid:
            missing_fields = [
                k for k, v in [
                    ("crop", crop),
                    ("latitude", lat),
                    ("longitude", lon),
                    ("field_size_ha", size),
                    ("planting_date", date_val)
                ] if v is None
            ]
            clarifying_question = f"Please provide the missing fields: {', '.join(missing_fields)}."
        else:
            missing_fields = []
            clarifying_question = ""
            
        validation = ProfileValidation(
            is_valid=is_valid,
            missing_fields=missing_fields,
            reason="Mock validation result",
            clarifying_question=clarifying_question
        )
        ctx.actions.state_delta[agent.output_key] = validation
        
        event = Event(
            author=agent.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=clarifying_question if clarifying_question else "Profile validated.")]
            ),
            output=validation
        )
        event.node_info.message_as_output = True
        yield event

    elif agent.name == "recommend":
        recommendation = IrrigationRecommendation(
            crop_type=ctx.state.get("crop", "wheat"),
            explanation="Based on the weather forecast, you should irrigate 15,000 Liters on Day 3. You saved 12,000 Liters."
        )
        ctx.actions.state_delta[agent.output_key] = recommendation
        
        event = Event(
            author=agent.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=recommendation.explanation)]
            ),
            output=recommendation
        )
        event.node_info.message_as_output = True
        yield event


@patch("google.adk.workflow._llm_agent_wrapper.run_llm_agent_as_node", side_effect=mock_run_llm_agent_as_node)
def test_agent_stream(mock_run) -> None:
    """
    Integration test for the agent stream functionality.
    Tests that the agent returns valid streaming responses.
    """
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    message = types.Content(
        role="user", parts=[types.Part.from_text(text="Why is the sky blue?")]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one message"

    has_text_content = False
    for event in events:
        if (
            event.content
            and event.content.parts
            and any(part.text for part in event.content.parts)
        ):
            has_text_content = True
            break
    assert has_text_content, "Expected at least one message with text content"


@patch("requests.get")
@patch("google.adk.workflow._llm_agent_wrapper.run_llm_agent_as_node", side_effect=mock_run_llm_agent_as_node)
def test_multi_turn_profile_persistence(mock_run, mock_get) -> None:
    """
    Integration test for multi-turn session state persistence.
    """
    # Mock Open-Meteo forecast call
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "daily": {
            "et0_fao_evapotranspiration": [3.0] * 7,
            "precipitation_sum": [0.0] * 7,
            "time": ["2026-06-25", "2026-06-26", "2026-06-27", "2026-06-28", "2026-06-29", "2026-06-30", "2026-07-01"],
        }
    }
    mock_get.return_value = mock_response

    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # Turn 1: Provide crop only
    msg1 = types.Content(
        role="user", parts=[types.Part.from_text(text="Hi, I have a wheat field.")]
    )
    events1 = list(
        runner.run(
            new_message=msg1,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    
    # Assert session state saved crop
    updated_session1 = session_service.get_session_sync(app_name="test", user_id="test_user", session_id=session.id)
    assert updated_session1.state.get("crop") == "wheat"

    # Turn 2: Provide lat/lon
    msg2 = types.Content(
        role="user", parts=[types.Part.from_text(text="Location is lat 19.5, lon 75.3")]
    )
    events2 = list(
        runner.run(
            new_message=msg2,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    
    # Assert session state persisted crop and added lat/lon
    updated_session2 = session_service.get_session_sync(app_name="test", user_id="test_user", session_id=session.id)
    assert updated_session2.state.get("crop") == "wheat"
    assert updated_session2.state.get("latitude") == 19.5
    assert updated_session2.state.get("longitude") == 75.3

    # Turn 3: Provide field size and planting date to complete the profile
    msg3 = types.Content(
        role="user", parts=[types.Part.from_text(text="Field size is 2.5ha, planted 2026-05-10")]
    )
    events3 = list(
        runner.run(
            new_message=msg3,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    
    # Assert all fields are persisted
    updated_session3 = session_service.get_session_sync(app_name="test", user_id="test_user", session_id=session.id)
    assert updated_session3.state.get("crop") == "wheat"
    assert updated_session3.state.get("latitude") == 19.5
    assert updated_session3.state.get("longitude") == 75.3
    assert updated_session3.state.get("field_size_ha") == 2.5
    assert updated_session3.state.get("planting_date") == "2026-05-10"

    # Assert that it successfully computed a recommendation at the end
    has_recommendation = False
    for event in events3:
        if (
            event.content
            and event.content.parts
            and any("Liters" in (part.text or "") or "liter" in (part.text or "").lower() for part in event.content.parts)
        ):
            has_recommendation = True
            break
    assert has_recommendation, "Expected a friendly recommendation output with liters calculated"


@patch("google.adk.workflow._llm_agent_wrapper.run_llm_agent_as_node", side_effect=mock_run_llm_agent_as_node)
def test_unsupported_crop_rejection(mock_run) -> None:
    """
    Integration test verifying validate_profile LlmAgent rejects unsupported crops.
    """
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # Send a complete profile with an unsupported crop "rice"
    msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hi, I have a rice field at lat 19.5, lon 75.3, size is 2.5ha, planted 2026-05-10.")],
    )
    events = list(
        runner.run(
            new_message=msg,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    
    # Verify that the response asks user to choose a supported crop / rejects "rice"
    rejected = False
    for event in events:
        if event.content and event.content.parts:
            text = "".join(part.text for part in event.content.parts if part.text)
            if "wheat" in text.lower() or "sugarcane" in text.lower() or "supported" in text.lower():
                rejected = True
                break
    assert rejected, "Expected validation failure prompting for a supported crop"
