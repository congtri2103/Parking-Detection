import unittest

from src.motion import MotionAnalyzer
from src.parking_state import ParkingStateMachine, ParkingStatus
from src.roi import bottom_center, parse_roi, point_in_polygon


class RoiTests(unittest.TestCase):
    def test_bottom_center_and_polygon(self):
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.assertEqual(bottom_center([10, 10, 30, 40]), (20.0, 40.0))
        self.assertTrue(point_in_polygon((50, 50), polygon))
        self.assertTrue(point_in_polygon((100, 50), polygon))
        self.assertFalse(point_in_polygon((101, 50), polygon))

    def test_parse_roi(self):
        self.assertEqual(parse_roi("0,0;10,0;10,10"), [(0, 0), (10, 0), (10, 10)])
        with self.assertRaises(ValueError):
            parse_roi("0,0;10,0")


class MotionTests(unittest.TestCase):
    def test_stationary_confirmation_and_movement(self):
        analyzer = MotionAnalyzer(5.0, confirmation_seconds=2.0, history_seconds=4.0)
        self.assertFalse(analyzer.update(7, (10, 10), 0.0).is_stopped)
        self.assertFalse(analyzer.update(7, (12, 10), 1.0).is_stopped)
        result = analyzer.update(7, (11, 11), 2.0)
        self.assertTrue(result.is_stopped)
        self.assertFalse(analyzer.update(7, (30, 10), 3.0).is_stopped)

    def test_non_monotonic_clock_resets_history(self):
        analyzer = MotionAnalyzer(5.0, confirmation_seconds=1.0, history_seconds=2.0)
        analyzer.update(1, (0, 0), 5.0)
        result = analyzer.update(1, (0, 0), 1.0)
        self.assertFalse(result.is_stopped)
        self.assertEqual(result.observed_seconds, 0.0)

    def test_irregular_frame_times_can_reach_confirmation(self):
        analyzer = MotionAnalyzer(5.0, confirmation_seconds=2.0, history_seconds=2.0)
        result = None
        for timestamp in (0.0, 0.7, 1.4, 2.1):
            result = analyzer.update(9, (10, 10), timestamp)
        self.assertIsNotNone(result)
        self.assertTrue(result.is_stopped)


class ParkingTests(unittest.TestCase):
    def test_violation_event_fires_once_and_resets(self):
        machine = ParkingStateMachine(parking_seconds=5.0)
        first = machine.update(3, is_stopped=True, inside_roi=True, timestamp=10.0)
        self.assertEqual(first.status, ParkingStatus.STOPPED)
        parked = machine.update(3, is_stopped=True, inside_roi=True, timestamp=15.0)
        self.assertTrue(parked.violation)
        self.assertTrue(parked.violation_started)
        repeated = machine.update(3, is_stopped=True, inside_roi=True, timestamp=16.0)
        self.assertTrue(repeated.violation)
        self.assertFalse(repeated.violation_started)
        moving = machine.update(3, is_stopped=False, inside_roi=True, timestamp=17.0)
        self.assertEqual(moving.status, ParkingStatus.MOVING)
        self.assertFalse(moving.violation)

    def test_stationary_outside_is_not_moving_or_violation(self):
        machine = ParkingStateMachine(parking_seconds=1.0)
        result = machine.update(4, is_stopped=True, inside_roi=False, timestamp=10.0)
        self.assertEqual(result.status, ParkingStatus.STOPPED)
        self.assertFalse(result.violation)


if __name__ == "__main__":
    unittest.main()
