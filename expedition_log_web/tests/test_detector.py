import unittest
from datetime import datetime, timedelta, timezone

from expedition_log_web.expedition_detector import Sample, detect_tack_jibe_events


class TestDetector(unittest.TestCase):
    def test_tack_detects_after_20s(self):
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        samples = []
        # 10s on +45
        for i in range(0, 10):
            samples.append(Sample(t_utc=t0 + timedelta(seconds=i), twa=45.0))
        # flip to -45 and keep it for 25s
        for i in range(10, 36):
            samples.append(Sample(t_utc=t0 + timedelta(seconds=i), twa=-45.0))

        events = detect_tack_jibe_events(samples)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "Tack")
        self.assertAlmostEqual(events[0].twa_before, 45.0, places=3)
        self.assertAlmostEqual(events[0].twa_after, -45.0, places=3)

        # Condition starts at t=10, reaches 20s at t=30 (inclusive sample at 30s).
        expected_event_utc = t0 + timedelta(seconds=30)
        expected_event_jst = expected_event_utc.astimezone(timezone(timedelta(hours=9)))
        self.assertEqual(events[0].time_jst, expected_event_jst)

    def test_jibe_has_priority_over_tack(self):
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        samples = []
        for i in range(0, 5):
            samples.append(Sample(t_utc=t0 + timedelta(seconds=i), twa=170.0))
        for i in range(5, 31):
            samples.append(Sample(t_utc=t0 + timedelta(seconds=i), twa=-170.0))

        events = detect_tack_jibe_events(samples)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "Jibe")


if __name__ == "__main__":
    unittest.main()

