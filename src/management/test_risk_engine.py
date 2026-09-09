import unittest
import math
from src.management.risk_engine import assess_risk

class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            'redLevel': 723.7, 'orangeLevel': 723.39, 'blueLevel': 721.56,
            'historical_95th_inflow': 500.0
        }

    # 1. Normal water level
    def test_normal_water_level(self):
        res = assess_risk({'waterLevel': 700.0}, self.metadata, 10.0, 10.0, 10.0)
        self.assertEqual(res['overall_status'], 'NORMAL')
        self.assertEqual(res['water_level_status'], 'NORMAL')
        self.assertEqual(res['inflow_forecast_status'], 'NORMAL')

    # 2. Blue threshold crossing
    def test_blue_threshold(self):
        res = assess_risk({'waterLevel': 722.0}, self.metadata, 10.0, 10.0, 10.0)
        self.assertEqual(res['overall_status'], 'WATCH')
        self.assertEqual(res['water_level_status'], 'WATCH')

    # 3. Orange threshold crossing
    def test_orange_threshold(self):
        res = assess_risk({'waterLevel': 723.5}, self.metadata, 10.0, 10.0, 10.0)
        self.assertEqual(res['overall_status'], 'ALERT')

    # 4. Red threshold crossing
    def test_red_threshold(self):
        res = assess_risk({'waterLevel': 724.0}, self.metadata, 10.0, 10.0, 10.0)
        self.assertEqual(res['overall_status'], 'HIGH RISK')

    # 5. High 1-day inflow
    def test_high_1d_inflow(self):
        res = assess_risk({'waterLevel': 700.0}, self.metadata, 600.0, 10.0, 10.0)
        self.assertEqual(res['inflow_forecast_status'], 'ALERT')
        self.assertEqual(res['overall_status'], 'ALERT')
        self.assertIn('1d', res['reason'])

    # 6. High 3-day inflow
    def test_high_3d_inflow(self):
        res = assess_risk({'waterLevel': 700.0}, self.metadata, 10.0, 600.0, 10.0)
        self.assertEqual(res['inflow_forecast_status'], 'ALERT')
        self.assertEqual(res['overall_status'], 'ALERT')
        self.assertIn('3d', res['reason'])

    # 7. High 7-day inflow
    def test_high_7d_inflow(self):
        res = assess_risk({'waterLevel': 700.0}, self.metadata, 10.0, 10.0, 600.0)
        self.assertEqual(res['inflow_forecast_status'], 'ALERT')
        self.assertEqual(res['overall_status'], 'ALERT')
        self.assertIn('7d', res['reason'])

    # 8. Multiple horizons triggering simultaneously
    def test_multiple_horizons(self):
        res = assess_risk({'waterLevel': 700.0}, self.metadata, 600.0, 600.0, 10.0)
        self.assertEqual(res['inflow_forecast_status'], 'ALERT')
        self.assertIn('1d', res['reason'])
        self.assertIn('3d', res['reason'])

    # 9. Missing water-level threshold
    def test_missing_wl_threshold(self):
        meta = {'historical_95th_inflow': 500.0} # no blue/orange/red
        res = assess_risk({'waterLevel': 750.0}, meta, 10.0, 10.0, 10.0)
        self.assertEqual(res['water_level_status'], 'NORMAL')

    # 10. Missing inflow history / Insufficient data (11)
    def test_missing_inflow_threshold(self):
        meta = {'redLevel': 723.7} # no historical_95th_inflow
        res = assess_risk({'waterLevel': 700.0}, meta, 600.0, 600.0, 600.0)
        self.assertEqual(res['inflow_forecast_status'], 'INSUFFICIENT_DATA')
        self.assertEqual(res['overall_status'], 'NORMAL')

    # 12. NaN/invalid forecast
    def test_nan_forecast(self):
        res = assess_risk({'waterLevel': 700.0}, self.metadata, None, 10.0, 10.0)
        self.assertEqual(res['inflow_forecast_status'], 'NORMAL')

    # 13. Negative forecast
    def test_negative_forecast(self):
        res = assess_risk({'waterLevel': 700.0}, self.metadata, -10.0, 10.0, 10.0)
        self.assertIn('Negative forecast detected', res['reason'])

    # 14. Missing reservoir (missing telemetry entirely)
    def test_missing_telemetry(self):
        res = assess_risk({}, self.metadata, 10.0, 10.0, 10.0)
        self.assertEqual(res['water_level_status'], 'UNKNOWN')

    # 15. Mixed current-level + high-inflow warning
    def test_mixed_warnings(self):
        res = assess_risk({'waterLevel': 722.0}, self.metadata, 600.0, 10.0, 10.0)
        self.assertEqual(res['water_level_status'], 'WATCH')
        self.assertEqual(res['inflow_forecast_status'], 'ALERT')
        self.assertEqual(res['overall_status'], 'ALERT')
        
if __name__ == '__main__':
    unittest.main()
