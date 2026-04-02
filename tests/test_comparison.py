import numpy as np
import pytest

from neuralplayground.comparison import PlaceCellScorer


def _make_gaussian_ratemap(nbins=50, center=None, sigma=5.0, peak=10.0):
    """Create a 2D Gaussian ratemap simulating a single place field."""
    if center is None:
        center = (nbins // 2, nbins // 2)
    x = np.arange(nbins)
    y = np.arange(nbins)
    xx, yy = np.meshgrid(x, y)
    rate_map = peak * np.exp(
        -((xx - center[1]) ** 2 + (yy - center[0]) ** 2) / (2 * sigma**2)
    )
    return rate_map


def _make_two_field_ratemap(nbins=50, sigma=4.0, peak=10.0):
    """Create a 2D ratemap with two spatially separated Gaussian place fields."""
    center_a = (nbins // 4, nbins // 4)
    center_b = (3 * nbins // 4, 3 * nbins // 4)
    rate_map = _make_gaussian_ratemap(nbins, center_a, sigma, peak)
    rate_map += _make_gaussian_ratemap(nbins, center_b, sigma, peak)
    return rate_map


@pytest.fixture
def scorer():
    return PlaceCellScorer(nbins=50)


class TestPlaceCellScorer:
    def test_init(self, scorer):
        assert isinstance(scorer, PlaceCellScorer)
        assert scorer._nbins == 50
        assert scorer._min_bins_per_field == 9
        assert scorer._field_threshold_fraction == 0.2

    def test_init_custom_params(self):
        s = PlaceCellScorer(
            nbins=100, min_bins_per_field=5, field_threshold_fraction=0.3
        )
        assert s._min_bins_per_field == 5
        assert s._field_threshold_fraction == 0.3

    # ------------------------------------------------------------------ #
    # spatial_information
    # ------------------------------------------------------------------ #

    def test_spatial_information_uniform_is_zero(self, scorer):
        """Uniform firing rate carries zero spatial information."""
        rate_map = np.ones((50, 50))
        info = scorer.spatial_information(rate_map)
        assert np.isclose(info, 0.0, atol=1e-10)

    def test_spatial_information_single_field_positive(self, scorer):
        """A peaked Gaussian ratemap should have positive spatial information."""
        rate_map = _make_gaussian_ratemap()
        info = scorer.spatial_information(rate_map)
        assert info > 0.0

    def test_spatial_information_all_nan(self, scorer):
        """All-NaN ratemap (no visited bins) should return NaN."""
        rate_map = np.full((50, 50), np.nan)
        info = scorer.spatial_information(rate_map)
        assert np.isnan(info)

    def test_spatial_information_zero_rate(self, scorer):
        """All-zero ratemap gives mean rate of zero, should return NaN."""
        rate_map = np.zeros((50, 50))
        info = scorer.spatial_information(rate_map)
        assert np.isnan(info)

    def test_spatial_information_with_occupancy(self, scorer):
        """Providing an explicit occupancy map should not raise and returns float."""
        rate_map = _make_gaussian_ratemap()
        occupancy = np.ones((50, 50))
        info = scorer.spatial_information(rate_map, occupancy_map=occupancy)
        assert isinstance(info, float)
        assert info > 0.0

    def test_spatial_information_with_nan_bins(self, scorer):
        """NaN bins are treated as unvisited and ignored in the computation."""
        rate_map = _make_gaussian_ratemap()
        rate_map[:5, :] = np.nan  # mark some bins as unvisited
        info = scorer.spatial_information(rate_map)
        assert isinstance(info, float)
        assert not np.isnan(info)

    # ------------------------------------------------------------------ #
    # detect_place_fields
    # ------------------------------------------------------------------ #

    def test_detect_single_field(self, scorer):
        """A single Gaussian peak should produce exactly one place field."""
        rate_map = _make_gaussian_ratemap()
        props = scorer.detect_place_fields(rate_map)
        assert props["n_fields"] == 1

    def test_detect_two_fields(self, scorer):
        """Two well-separated Gaussian peaks should yield two place fields."""
        rate_map = _make_two_field_ratemap()
        props = scorer.detect_place_fields(rate_map)
        assert props["n_fields"] == 2

    def test_detect_no_fields_nan(self, scorer):
        """All-NaN ratemap should return zero fields."""
        rate_map = np.full((50, 50), np.nan)
        props = scorer.detect_place_fields(rate_map)
        assert props["n_fields"] == 0
        assert props["field_sizes_bins"] == []
        assert props["peak_coords"] == []

    def test_detect_no_fields_zero(self, scorer):
        """All-zero ratemap should return zero fields."""
        rate_map = np.zeros((50, 50))
        props = scorer.detect_place_fields(rate_map)
        assert props["n_fields"] == 0

    def test_detect_field_props_types(self, scorer):
        """Check that field properties have the expected types."""
        rate_map = _make_gaussian_ratemap()
        props = scorer.detect_place_fields(rate_map)
        assert isinstance(props["n_fields"], int)
        assert isinstance(props["field_sizes_bins"], list)
        assert isinstance(props["field_size_mean_fraction"], float)
        assert isinstance(props["peak_firing_rate"], float)
        assert isinstance(props["mean_firing_rate"], float)
        assert isinstance(props["peak_coords"], list)
        assert isinstance(props["field_labels"], np.ndarray)

    def test_detect_field_labels_shape(self, scorer):
        """field_labels must have the same shape as the input ratemap."""
        rate_map = _make_gaussian_ratemap(nbins=50)
        props = scorer.detect_place_fields(rate_map)
        assert props["field_labels"].shape == rate_map.shape

    def test_detect_peak_coords_within_field(self, scorer):
        """Each peak coordinate should lie inside its corresponding field."""
        rate_map = _make_gaussian_ratemap()
        props = scorer.detect_place_fields(rate_map)
        assert len(props["peak_coords"]) == 1
        row, col = props["peak_coords"][0]
        assert props["field_labels"][row, col] == 1

    def test_detect_field_size_fraction_in_range(self, scorer):
        """Mean field size fraction should be between 0 and 1."""
        rate_map = _make_gaussian_ratemap()
        props = scorer.detect_place_fields(rate_map)
        assert 0.0 < props["field_size_mean_fraction"] < 1.0

    # ------------------------------------------------------------------ #
    # get_scores
    # ------------------------------------------------------------------ #

    def test_get_scores_returns_all_keys(self, scorer):
        expected_keys = {
            "spatial_information",
            "n_fields",
            "field_sizes_bins",
            "field_size_mean_fraction",
            "peak_firing_rate",
            "mean_firing_rate",
            "peak_coords",
            "field_labels",
        }
        rate_map = _make_gaussian_ratemap()
        scores = scorer.get_scores(rate_map)
        assert expected_keys == set(scores.keys())

    def test_get_scores_single_field(self, scorer):
        """Single Gaussian ratemap: spatial info > 0 and exactly one field."""
        rate_map = _make_gaussian_ratemap()
        scores = scorer.get_scores(rate_map)
        assert scores["spatial_information"] > 0.0
        assert scores["n_fields"] == 1

    def test_get_scores_with_occupancy(self, scorer):
        """get_scores passes occupancy_map through to spatial_information."""
        rate_map = _make_gaussian_ratemap()
        occupancy = np.ones((50, 50))
        scores = scorer.get_scores(rate_map, occupancy_map=occupancy)
        assert isinstance(scores["spatial_information"], float)
