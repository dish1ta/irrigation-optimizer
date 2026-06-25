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
from unittest.mock import MagicMock, patch
from datetime import date

from google.genai import types
from pydantic import ValidationError

from app.agent import FORECAST_CACHE, fetch_weather, get_forecast


class TestGetForecast(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test
        FORECAST_CACHE.clear()

    @patch("requests.get")
    def test_normal_response(self, mock_get):
        # Mock normal Open-Meteo response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "daily": {
                "et0_fao_evapotranspiration": [3.2, 4.0, 2.5, 3.8, 4.2, 3.5, 3.1],
                "precipitation_sum": [0.0, 1.2, 0.0, 0.0, 5.4, 0.0, 0.0],
            }
        }
        mock_get.return_value = mock_response

        # Execute
        results = get_forecast(lat=-1.2921, lon=36.8219, days=7)

        # Asserts
        self.assertEqual(len(results), 7)
        self.assertEqual(results[0]["day"], 1)
        self.assertEqual(results[0]["et0_mm"], 3.2)
        self.assertEqual(results[0]["precipitation_mm"], 0.0)

        # Call again to verify cache hit (requests.get called only once)
        cached_results = get_forecast(lat=-1.2921, lon=36.8219, days=7)
        self.assertEqual(cached_results, results)
        mock_get.assert_called_once()

    def test_coordinate_range_validation(self):
        # Test out of bounds latitude
        with self.assertRaises(ValueError) as ctx:
            get_forecast(lat=95.0, lon=120.0)
        self.assertIn("Latitude 95.0 is out of bounds", str(ctx.exception))

        # Test out of bounds longitude
        with self.assertRaises(ValueError) as ctx:
            get_forecast(lat=10.0, lon=-185.0)
        self.assertIn("Longitude -185.0 is out of bounds", str(ctx.exception))

    @patch("requests.get")
    def test_malformed_response_missing_keys(self, mock_get):
        # Mock malformed response missing daily key
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"hourly": {}}
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            get_forecast(lat=0.0, lon=0.0)
        self.assertIn("'daily' section is missing", str(ctx.exception))

        # Mock response missing specific daily columns
        mock_response.json.return_value = {"daily": {"time": []}}
        with self.assertRaises(ValueError) as ctx:
            get_forecast(lat=0.0, lon=0.0)
        self.assertIn("are missing", str(ctx.exception))

    @patch("requests.get")
    def test_malformed_response_non_numeric(self, mock_get):
        # Mock malformed response with non-numeric value (string)
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "daily": {
                "et0_fao_evapotranspiration": [3.2, "four", 2.5, 3.8, 4.2, 3.5, 3.1],
                "precipitation_sum": [0.0, 1.2, 0.0, 0.0, 5.4, 0.0, 0.0],
            }
        }
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            get_forecast(lat=0.0, lon=0.0)
        self.assertIn("non-numeric daily value at index 1", str(ctx.exception))

    @patch("requests.get")
    def test_malformed_response_boolean(self, mock_get):
        # Mock malformed response with boolean values (since True/False are subclasses of int)
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "daily": {
                "et0_fao_evapotranspiration": [3.2, True, 2.5, 3.8, 4.2, 3.5, 3.1],
                "precipitation_sum": [0.0, 1.2, 0.0, 0.0, 5.4, 0.0, 0.0],
            }
        }
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            get_forecast(lat=0.0, lon=0.0)
        self.assertIn("non-numeric daily value at index 1", str(ctx.exception))

    @patch("requests.get")
    def test_caching_ttl(self, mock_get):
        # Setup mock
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "daily": {
                "et0_fao_evapotranspiration": [3.2] * 7,
                "precipitation_sum": [0.0] * 7,
            }
        }
        mock_get.return_value = mock_response

        # Execute call 1
        get_forecast(lat=10.0, lon=20.0, days=7)
        self.assertEqual(mock_get.call_count, 1)

        # Modify the timestamp in the cache to make it expired (older than 1 hour)
        cache_key = (10.0, 20.0, 7)
        timestamp, data = FORECAST_CACHE[cache_key]
        FORECAST_CACHE[cache_key] = (timestamp - 3601, data)

        # Execute call 2 (expires -> triggers requests.get again)
        get_forecast(lat=10.0, lon=20.0, days=7)
        self.assertEqual(mock_get.call_count, 2)


class TestFetchWeather(unittest.TestCase):
    def setUp(self):
        FORECAST_CACHE.clear()

    @patch("requests.get")
    def test_fetch_weather_success(self, mock_get):
        # Mock successful forecast response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "daily": {
                "et0_fao_evapotranspiration": [3.2] * 7,
                "precipitation_sum": [0.0] * 7,
            }
        }
        mock_get.return_value = mock_response

        # Test with coordinates stored in state
        ctx = MagicMock()
        ctx.state = {"latitude": -1.2921, "longitude": 36.8219}

        # Call the underlying function of fetch_weather
        result = fetch_weather._func(ctx, None)

        # Verify coordinates and results
        self.assertEqual(len(result.forecast), 7)
        self.assertEqual(result.forecast[0].et0_mm, 3.2)
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["latitude"], -1.2921)
        self.assertEqual(kwargs["params"]["longitude"], 36.8219)

    def test_fetch_weather_missing_state_coords(self):
        # Verify KeyError is raised when coordinates are missing in state (no fallback to default Nairobi coords)
        ctx = MagicMock()
        ctx.state = {}
        with self.assertRaises(KeyError):
            fetch_weather._func(ctx, None)

    @patch("app.agent.get_forecast", side_effect=ValueError("Weather API is down or returned malformed data"))
    def test_fetch_weather_propagates_api_error(self, mock_get_forecast):
        # Verify that get_forecast errors propagate directly and raise instead of silently swallowing or falling back
        ctx = MagicMock()
        ctx.state = {"latitude": 1.29, "longitude": 36.82}
        with self.assertRaises(ValueError) as context:
            fetch_weather._func(ctx, None)
        self.assertEqual(str(context.exception), "Weather API is down or returned malformed data")
