"""Place cell metrics for scoring spatial representations.

References
----------
Skaggs WE, McNaughton BL, Gothard KM, Markus EJ (1993).
An information-theoretic approach to deciphering the hippocampal code.
Advances in Neural Information Processing Systems, 5, 1030-1037.

O'Keefe J, Dostrovsky J (1971).
The hippocampus as a spatial map. Preliminary evidence from unit activity
in the freely-moving rat. Brain Research, 34(1):171-5.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from neuralplayground.plotting.plot_utils import make_plot_rate_map


class PlaceCellScorer(object):
    """Class for scoring ratemaps to assess place cell properties.

    Computes the spatial information score (Skaggs et al., 1993) and detects
    place fields from a 2D spatial ratemap, returning metrics including number
    of fields, field sizes, peak firing rate and spatial information.

    References
    ----------
    Skaggs WE, McNaughton BL, Gothard KM, Markus EJ (1993).
    An information-theoretic approach to deciphering the hippocampal code.
    Advances in Neural Information Processing Systems, 5, 1030-1037.

    O'Keefe J, Dostrovsky J (1971).
    The hippocampus as a spatial map. Preliminary evidence from unit activity
    in the freely-moving rat. Brain Research, 34(1):171-5.

    """

    def __init__(self, nbins, min_bins_per_field=9, field_threshold_fraction=0.2):
        """Initialise PlaceCellScorer.

        Parameters
        ----------
        nbins : int
            Number of bins per dimension in the ratemap.
        min_bins_per_field : int, optional
            Minimum number of contiguous bins required for a region to be
            counted as a place field. Default 9.
        field_threshold_fraction : float, optional
            Fraction of the peak firing rate used as the lower boundary
            threshold when detecting place fields. Default 0.2.

        """
        self._nbins = nbins
        self._min_bins_per_field = min_bins_per_field
        self._field_threshold_fraction = field_threshold_fraction

    def spatial_information(self, rate_map, occupancy_map=None):
        """Compute the spatial information score (Skaggs et al., 1993).

        Measures the amount of information (in bits per spike) that a single
        spike conveys about the animal's spatial location.

        Parameters
        ----------
        rate_map : array_like
            2D array of firing rates per spatial bin. NaN values indicate
            unvisited bins.
        occupancy_map : array_like, optional
            2D array of time spent in each spatial bin, same shape as
            rate_map. If None, uniform occupancy is assumed across all
            visited bins.

        Returns
        -------
        spatial_info : float
            Spatial information in bits per spike. Returns NaN when no bins
            are visited or the mean firing rate is zero.

        Notes
        -----
        The spatial information score is:

        I = sum_i p_i * (lambda_i / lambda_mean) * log2(lambda_i / lambda_mean)

        where p_i is the occupancy probability of bin i, lambda_i is the
        firing rate in bin i, and lambda_mean is the overall mean firing rate
        weighted by occupancy.

        """
        rate_map = np.asarray(rate_map, dtype=float)
        visited_mask = np.isfinite(rate_map)
        n_visited = int(np.sum(visited_mask))

        if n_visited == 0:
            return np.nan

        if occupancy_map is None:
            occupancy = np.where(visited_mask, 1.0 / n_visited, 0.0)
        else:
            occupancy_map = np.asarray(occupancy_map, dtype=float)
            total_time = np.nansum(occupancy_map)
            if total_time == 0:
                return np.nan
            occupancy = np.where(visited_mask, occupancy_map / total_time, 0.0)

        rates = np.where(visited_mask, rate_map, 0.0)
        mean_rate = np.sum(occupancy * rates)

        if mean_rate == 0:
            return np.nan

        # Bins with zero rate contribute 0 to the sum (limit of x*log2(x) = 0)
        nonzero_mask = visited_mask & (rates > 0)
        rate_ratio = np.where(nonzero_mask, rates / mean_rate, 1.0)
        log_term = np.where(nonzero_mask, np.log2(rate_ratio), 0.0)
        spatial_info = float(np.sum(occupancy * rate_ratio * log_term))

        return spatial_info

    def detect_place_fields(self, rate_map):
        """Detect place fields from a 2D firing rate map.

        A place field is a contiguous region of bins whose firing rate exceeds
        a threshold fraction of the map's peak rate. Regions smaller than
        ``min_bins_per_field`` are discarded.

        Parameters
        ----------
        rate_map : array_like
            2D array of firing rates per spatial bin. NaN values indicate
            unvisited bins.

        Returns
        -------
        field_props : dict
            Dictionary containing:

            * n_fields : int
                Number of detected place fields after size filtering.
            * field_sizes_bins : list of int
                Number of bins in each detected field.
            * field_size_mean_fraction : float
                Mean field size as a fraction of total visited bins.
                NaN if no fields are detected.
            * peak_firing_rate : float
                Maximum firing rate across all visited bins.
            * mean_firing_rate : float
                Mean firing rate across all visited bins.
            * peak_coords : list of tuple of int
                (row, col) coordinates of the peak firing rate bin within
                each detected field.
            * field_labels : np.ndarray
                2D integer array with a unique positive label per detected
                field. Zero indicates bins not belonging to any field.

        Notes
        -----
        Field boundaries are set at
        ``field_threshold_fraction * peak_firing_rate``.

        """
        rate_map = np.asarray(rate_map, dtype=float)
        visited_mask = np.isfinite(rate_map)
        n_visited = int(np.sum(visited_mask))

        peak_firing_rate = float(np.nanmax(rate_map)) if n_visited > 0 else np.nan
        mean_firing_rate = (
            float(np.nanmean(rate_map[visited_mask])) if n_visited > 0 else np.nan
        )

        empty_result = {
            "n_fields": 0,
            "field_sizes_bins": [],
            "field_size_mean_fraction": np.nan,
            "peak_firing_rate": peak_firing_rate,
            "mean_firing_rate": mean_firing_rate,
            "peak_coords": [],
            "field_labels": np.zeros(rate_map.shape, dtype=int),
        }

        if n_visited == 0 or np.isnan(peak_firing_rate) or peak_firing_rate == 0:
            return empty_result

        threshold = self._field_threshold_fraction * peak_firing_rate
        above_threshold = visited_mask & (rate_map >= threshold)
        labeled, n_raw_fields = ndimage.label(above_threshold)

        valid_labels = []
        field_sizes = []
        peak_coords = []

        for label_id in range(1, n_raw_fields + 1):
            field_mask = labeled == label_id
            field_size = int(np.sum(field_mask))
            if field_size >= self._min_bins_per_field:
                valid_labels.append(label_id)
                field_sizes.append(field_size)
                field_rates = np.where(field_mask, rate_map, -np.inf)
                peak_idx = np.unravel_index(np.argmax(field_rates), rate_map.shape)
                peak_coords.append(peak_idx)

        field_labels = np.zeros(rate_map.shape, dtype=int)
        for new_id, old_label in enumerate(valid_labels, start=1):
            field_labels[labeled == old_label] = new_id

        n_fields = len(valid_labels)
        field_size_mean_fraction = (
            float(np.mean(field_sizes) / n_visited) if n_fields > 0 else np.nan
        )

        return {
            "n_fields": n_fields,
            "field_sizes_bins": field_sizes,
            "field_size_mean_fraction": field_size_mean_fraction,
            "peak_firing_rate": peak_firing_rate,
            "mean_firing_rate": mean_firing_rate,
            "peak_coords": peak_coords,
            "field_labels": field_labels,
        }

    def get_scores(self, rate_map, occupancy_map=None):
        """Compute all place cell scores for a given ratemap.

        Parameters
        ----------
        rate_map : np.ndarray
            2D array of firing rates per spatial bin. NaN values indicate
            unvisited bins.
        occupancy_map : np.ndarray, optional
            2D array of time spent in each spatial bin, same shape as
            rate_map. If None, uniform occupancy is assumed across all
            visited bins.

        Returns
        -------
        scores : dict
            Dictionary containing:

            * spatial_information : float
                Spatial information in bits per spike (Skaggs et al., 1993).
            * n_fields : int
                Number of detected place fields.
            * field_sizes_bins : list of int
                Number of bins in each detected field.
            * field_size_mean_fraction : float
                Mean field size as a fraction of total visited area.
            * peak_firing_rate : float
                Maximum firing rate across all visited bins.
            * mean_firing_rate : float
                Mean firing rate across all visited bins.
            * peak_coords : list of tuple of int
                (row, col) coordinates of the peak in each detected field.
            * field_labels : np.ndarray
                2D integer array labelling each detected field.

        See Also
        --------
        PlaceCellScorer.spatial_information : Skaggs spatial information score.
        PlaceCellScorer.detect_place_fields : Place field detection.

        """
        scores = {
            "spatial_information": self.spatial_information(rate_map, occupancy_map)
        }
        scores.update(self.detect_place_fields(rate_map))
        return scores

    def plot_place_fields(self, rate_map, ax=None, title="Place fields"):
        """Plot the ratemap with detected place field boundaries overlaid.

        Displays the firing rate map using the standard ratemap colormap and
        draws white contours around each detected place field, with a cross
        marking each field's peak location.

        Parameters
        ----------
        rate_map : np.ndarray
            2D array of firing rates per spatial bin.
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw. Uses the current axes if None.
        title : str, optional
            Plot title. Default "Place fields".

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the plot was drawn on.

        """
        if ax is None:
            ax = plt.gca()

        ax = make_plot_rate_map(
            rate_map, ax, title, "width", "depth", "firing rate (Hz)"
        )

        field_props = self.detect_place_fields(rate_map)
        for field_id in range(1, field_props["n_fields"] + 1):
            field_mask = (field_props["field_labels"] == field_id).astype(float)
            ax.contour(field_mask, levels=[0.5], colors="white", linewidths=1.5)

        for peak_row, peak_col in field_props["peak_coords"]:
            ax.plot(peak_col, peak_row, "w+", markersize=8, markeredgewidth=2)

        return ax
