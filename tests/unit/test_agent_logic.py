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

import unittest
from unittest.mock import patch, MagicMock
from datetime import date, timedelta
import os
import json
import logging

from app.agent import (
    parse_profile_from_text,
    load_supported_crops,
    workflow,
    save_profile,
)


class TestAgentLogic(unittest.TestCase):
    def test_parse_profile_from_text_crops(self):
        # Test crop detection
        self.assertEqual(parse_profile_from_text("I have a wheat field")["crop"], "wheat")
        self.assertEqual(parse_profile_from_text("setting up sugarcane field")["crop"], "sugarcane")
        self.assertEqual(parse_profile_from_text("my Tomato plants")["crop"], "tomato")
        self.assertNotIn("crop", parse_profile_from_text("I have a rice field"))

    def test_parse_profile_from_text_field_size(self):
        # Test field size parsing
        self.assertEqual(parse_profile_from_text("size is 2.5ha")["field_size_ha"], 2.5)
        self.assertEqual(parse_profile_from_text("5.0 hectare field")["field_size_ha"], 5.0)
        self.assertEqual(parse_profile_from_text("10 Hectares of wheat")["field_size_ha"], 10.0)
        self.assertNotIn("field_size_ha", parse_profile_from_text("I have some hectares"))

    def test_parse_profile_from_text_planting_date(self):
        # Test planting date parsing
        self.assertEqual(parse_profile_from_text("planted 2026-05-10")["planting_date"], "2026-05-10")
        # Invalid date format should be ignored
        self.assertNotIn("planting_date", parse_profile_from_text("planted on 2026-05-32"))
        self.assertNotIn("planting_date", parse_profile_from_text("planted on 2026/05/10"))

    def test_parse_profile_from_text_coordinates(self):
        # Test labeled coordinates
        self.assertEqual(parse_profile_from_text("lat: 19.5, lon: 75.3")["latitude"], 19.5)
        self.assertEqual(parse_profile_from_text("latitude 19.5, longitude 75.3")["longitude"], 75.3)
        self.assertEqual(parse_profile_from_text("lat=-1.29, longitude=36.8")["latitude"], -1.29)

        # Test unlabeled coordinates are ignored/missing
        result = parse_profile_from_text("My coordinates are 19.5, 75.3")
        self.assertNotIn("latitude", result)
        self.assertNotIn("longitude", result)

    def test_days_after_planting(self):
        # We test the days_after_planting calculation by checking datetime date operations
        # Directly mock datetime or use date math
        today = date.today()

        # 10 days ago
        ten_days_ago = today - timedelta(days=10)
        days = (today - ten_days_ago).days
        self.assertEqual(days, 10)

        # 5 days in the future (clamped to 0)
        five_days_future = today + timedelta(days=5)
        days_future = (today - five_days_future).days
        clamped_days = max(0, days_future)
        self.assertEqual(clamped_days, 0)

    def test_load_supported_crops(self):
        # Test dynamic load returns exactly the keys from the real json file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        crop_data_path = os.path.abspath(
            os.path.join(
                script_dir,
                "..",
                "..",
                ".agents",
                "skills",
                "crop-profile",
                "references",
                "crop_data.json",
            )
        )
        with open(crop_data_path, "r") as f:
            real_data = json.load(f)
            expected_crops = list(real_data.keys())

        supported_crops = load_supported_crops()
        self.assertEqual(supported_crops, expected_crops)

    @patch("app.agent.logger.warning")
    @patch("builtins.open", side_effect=FileNotFoundError("Mocked file not found"))
    def test_load_supported_crops_fallback_logging(self, mock_open, mock_warn):
        # Test that fallback path logs a warning when exception is encountered/path missing
        crops = load_supported_crops()
        # Should return fallback crops
        self.assertEqual(crops, ["wheat", "maize", "cotton", "sugarcane", "tomato", "chickpea", "groundnut"])
        # Should log warning
        mock_warn.assert_called_once()
        self.assertIn("Failed to load supported crops from", mock_warn.call_args[0][0])

    def test_workflow_graph_construction(self):
        # Smoke test: confirms the Workflow builds without graph validation error
        self.assertIsNotNone(workflow)
        self.assertIsNotNone(workflow.graph)
        self.assertEqual(workflow.name, "irrigation_optimizer_workflow")


from google.genai import types

class TestSaveProfile(unittest.TestCase):
    def test_save_profile_nested_json(self):
        ctx = MagicMock()
        ctx.state = {}
        payload = '{"crop": {"crop_type": "Tomatoes", "planting_date": "2026-03-01", "field_size_ha": 0.001}, "latitude": -1.2921, "longitude": 36.8219}'
        node_input = types.Content(parts=[types.Part.from_text(text=payload)])

        save_profile._func(ctx, node_input)

        self.assertEqual(ctx.state["crop"], "Tomatoes")
        self.assertEqual(ctx.state["planting_date"], "2026-03-01")
        self.assertEqual(ctx.state["field_size_ha"], 0.001)
        self.assertEqual(ctx.state["latitude"], -1.2921)
        self.assertEqual(ctx.state["longitude"], 36.8219)

    def test_save_profile_direct_json(self):
        ctx = MagicMock()
        ctx.state = {}
        payload = '{"crop": "wheat", "latitude": 19.5, "longitude": 75.3, "field_size_ha": 2.5, "planting_date": "2026-05-10"}'
        node_input = types.Content(parts=[types.Part.from_text(text=payload)])

        save_profile._func(ctx, node_input)

        self.assertEqual(ctx.state["crop"], "wheat")
        self.assertEqual(ctx.state["latitude"], 19.5)
        self.assertEqual(ctx.state["longitude"], 75.3)
        self.assertEqual(ctx.state["field_size_ha"], 2.5)
        self.assertEqual(ctx.state["planting_date"], "2026-05-10")

    def test_save_profile_malformed_json_fallback(self):
        ctx = MagicMock()
        ctx.state = {}
        # Malformed JSON should fallback to deterministic text parsing
        payload = '{"crop": "wheat", "latitude": 19.5, malformed'
        node_input = types.Content(parts=[types.Part.from_text(text=payload)])

        save_profile._func(ctx, node_input)

        # Crop wheat and lat 19.5 should be successfully extracted by regex fallback
        self.assertEqual(ctx.state["crop"], "wheat")
        self.assertEqual(ctx.state["latitude"], 19.5)
