# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for StreamingGrouper class."""

import unittest

from cutracer.analysis import StreamingGrouper


class TestStreamingGrouperInit(unittest.TestCase):
    """Tests for StreamingGrouper initialization."""

    def test_init_with_iterator(self):
        """Test initialization with an iterator."""
        records = iter([{"warp": 0}, {"warp": 1}])
        grouper = StreamingGrouper(records, "warp")
        self.assertIsNotNone(grouper)

    def test_init_with_list_iterator(self):
        """Test initialization converts list usage correctly."""
        records = [{"warp": 0}, {"warp": 1}]
        grouper = StreamingGrouper(iter(records), "warp")
        groups = grouper.tail_per_group(10)
        self.assertEqual(len(groups), 2)


class TestStreamingGrouperTailPerGroup(unittest.TestCase):
    """Tests for tail_per_group method."""

    def test_tail_per_group_basic(self):
        """Test basic tail_per_group functionality."""
        records = iter([
            {"warp": 0, "pc": 0},
            {"warp": 0, "pc": 16},
            {"warp": 0, "pc": 32},
            {"warp": 1, "pc": 0},
            {"warp": 1, "pc": 16},
        ])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(2)

        # Should have 2 groups
        self.assertEqual(len(groups), 2)
        # Warp 0 should have last 2 records
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[0][0]["pc"], 16)
        self.assertEqual(groups[0][1]["pc"], 32)
        # Warp 1 should have last 2 records
        self.assertEqual(len(groups[1]), 2)

    def test_tail_per_group_more_than_available(self):
        """Test tail_per_group when requesting more than available."""
        records = iter([
            {"warp": 0, "pc": 0},
            {"warp": 0, "pc": 16},
        ])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(10)

        # Should return all available records
        self.assertEqual(len(groups[0]), 2)

    def test_tail_per_group_empty(self):
        """Test tail_per_group with empty iterator."""
        records = iter([])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(10)

        self.assertEqual(groups, {})

    def test_tail_per_group_zero(self):
        """Test tail_per_group with n=0."""
        records = iter([{"warp": 0, "pc": 0}])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(0)

        self.assertEqual(groups, {})


class TestStreamingGrouperHeadPerGroup(unittest.TestCase):
    """Tests for head_per_group method."""

    def test_head_per_group_basic(self):
        """Test basic head_per_group functionality."""
        records = iter([
            {"warp": 0, "pc": 0},
            {"warp": 0, "pc": 16},
            {"warp": 0, "pc": 32},
            {"warp": 1, "pc": 0},
            {"warp": 1, "pc": 16},
        ])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.head_per_group(2)

        # Should have 2 groups
        self.assertEqual(len(groups), 2)
        # Warp 0 should have first 2 records
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[0][0]["pc"], 0)
        self.assertEqual(groups[0][1]["pc"], 16)
        # Warp 1 should have first 2 records
        self.assertEqual(len(groups[1]), 2)

    def test_head_per_group_more_than_available(self):
        """Test head_per_group when requesting more than available."""
        records = iter([
            {"warp": 0, "pc": 0},
            {"warp": 0, "pc": 16},
        ])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.head_per_group(10)

        # Should return all available records
        self.assertEqual(len(groups[0]), 2)


class TestStreamingGrouperCountPerGroup(unittest.TestCase):
    """Tests for count_per_group method."""

    def test_count_per_group_basic(self):
        """Test basic count_per_group functionality."""
        records = iter([
            {"warp": 0, "pc": 0},
            {"warp": 0, "pc": 16},
            {"warp": 0, "pc": 32},
            {"warp": 1, "pc": 0},
            {"warp": 1, "pc": 16},
        ])
        grouper = StreamingGrouper(records, "warp")
        counts = grouper.count_per_group()

        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 2)


class TestStreamingGrouperConsumed(unittest.TestCase):
    """Tests for consumed state handling."""

    def test_cannot_use_twice(self):
        """Test that grouper raises error when used twice."""
        records = iter([{"warp": 0}])
        grouper = StreamingGrouper(records, "warp")

        # First call should work
        grouper.tail_per_group(10)

        # Second call should raise
        with self.assertRaises(RuntimeError) as ctx:
            grouper.tail_per_group(10)

        self.assertIn("already been consumed", str(ctx.exception))


class TestStreamingGrouperGroupByField(unittest.TestCase):
    """Tests for grouping by different fields."""

    def test_group_by_string_field(self):
        """Test grouping by a string field (sass)."""
        records = iter([
            {"sass": "ADD R0, R1", "pc": 0},
            {"sass": "ADD R0, R1", "pc": 16},
            {"sass": "MUL R2, R3", "pc": 32},
        ])
        grouper = StreamingGrouper(records, "sass")
        groups = grouper.tail_per_group(10)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups["ADD R0, R1"]), 2)
        self.assertEqual(len(groups["MUL R2, R3"]), 1)

    def test_group_by_missing_field(self):
        """Test grouping when field is missing in some records."""
        records = iter([
            {"warp": 0, "pc": 0},
            {"pc": 16},  # Missing warp
            {"warp": 1, "pc": 32},
        ])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(10)

        # Should have 3 groups: 0, 1, and None
        self.assertEqual(len(groups), 3)
        self.assertIn(None, groups)


if __name__ == "__main__":
    unittest.main()
