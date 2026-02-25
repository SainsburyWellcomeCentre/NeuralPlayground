"""
Spatial Information Rate metric based on Skaggs et al. (1992).

Reference
---------
Skaggs, W. E., McNaughton, B. L., Gothard, K. M., & Markus, E. J. (1992).
An information-theoretic approach to deciphering the hippocampal code.
Advances in Neural Information Processing Systems (NIPS), 1030-1037.

The core measure is the information rate of a neuron:

    I = ∑_x  p(x) · λ(x) · log₂[ λ(x) / λ ]   (bits / time-step)

where
    x       = spatial bin
    p(x)    = probability (fraction of time) the agent occupies bin x
    λ(x)    = mean activation when the agent is at bin x
    λ       = overall mean activation  =  ∑_x p(x) · λ(x)

Dividing I by λ gives information per spike (bits / spike), a measure of
spatial specificity that is independent of mean firing rate.

Compatibility with NeuralPlayground agents
------------------------------------------
``InformationRateScorer.compute_from_ratemap`` is the general entry-point.
It accepts any 2-D ratemap and matching occupancy map,
so it works with any agent that exposes a spatial firing-rate map.

For ``George2021`` (CSCG/CHMM) agents a dedicated helper
``compute_from_cscg_agent`` extracts per-clone place-field ratemaps from the
agent's belief states and computes an information rate for each clone.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


class InformationRateScorer:
    """Compute the Skaggs et al. (1992) spatial information rate.

    Parameters
    ----------
    n_bins_x : int
        Number of spatial bins along the x-axis when building occupancy /
        ratemap grids from raw position data.
    n_bins_y : int
        Number of spatial bins along the y-axis.  Defaults to ``n_bins_x``.
    smoothing_sigma : float
        Gaussian smoothing applied to occupancy and rate maps before the
        information calculation.  Set to 0.0 to disable.
    min_occupancy : float
        Bins with occupancy below this threshold are treated as unvisited and
        excluded from the information calculation (avoids division by zero and
        unreliable rate estimates).
    """

    def __init__(
        self,
        n_bins_x: int = 20,
        n_bins_y: Optional[int] = None,
        smoothing_sigma: float = 1.0,
        min_occupancy: float = 1e-8,
    ):
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y if n_bins_y is not None else n_bins_x
        self.smoothing_sigma = smoothing_sigma
        self.min_occupancy = min_occupancy

    def compute_from_ratemap(
        self,
        ratemap: np.ndarray,
        occupancy: np.ndarray,
    ) -> dict:
        """Compute information rate from a pre-built ratemap and occupancy map.

        Both arrays must have the same shape ``(n_bins_y, n_bins_x)``.

        Parameters
        ----------
        ratemap : ndarray, shape (n_bins_y, n_bins_x)
            Mean activation (firing rate, belief-state probability, …) of a
            single unit in each spatial bin.
        occupancy : ndarray, shape (n_bins_y, n_bins_x)
            Number of time steps spent in each bin (raw counts or dwell time).
            Need not be normalised – the scorer normalises internally.

        Returns
        -------
        dict with keys:
            ``information_rate``  – I in bits per time-step
            ``information_per_spike`` – I / λ in bits per activation unit
            ``mean_rate``          – λ, the overall mean activation
            ``ratemap``            – the (smoothed) ratemap used
            ``occupancy_prob``     – the normalised occupancy map p(x)
        """
        ratemap = np.asarray(ratemap, dtype=float)
        occupancy = np.asarray(occupancy, dtype=float)

        if ratemap.shape != occupancy.shape:
            raise ValueError(f"ratemap shape {ratemap.shape} and occupancy shape {occupancy.shape} must match.")

        # Optional smoothing
        if self.smoothing_sigma > 0:
            ratemap = gaussian_filter(ratemap, sigma=self.smoothing_sigma)
            occupancy = gaussian_filter(occupancy, sigma=self.smoothing_sigma)

        # Normalise occupancy to probability
        total_occ = occupancy.sum()
        if total_occ == 0:
            raise ValueError("Occupancy map is all-zero; no position data found.")
        p_x = occupancy / total_occ  # shape (ny, nx)

        # Mask unvisited / low-occupancy bins
        visited = occupancy >= self.min_occupancy

        # Overall mean activation  λ = Σ_x p(x)·λ(x)
        lambda_mean = (p_x[visited] * ratemap[visited]).sum()

        if lambda_mean == 0:
            warnings.warn(
                "Mean activation is zero – information rate is undefined. Returning 0 for both measures.",
                RuntimeWarning,
            )
            return {
                "information_rate": 0.0,
                "information_per_spike": 0.0,
                "mean_rate": 0.0,
                "ratemap": ratemap,
                "occupancy_prob": p_x,
            }

        # Skaggs formula  I = Σ_x p(x)·λ(x)·log2[λ(x)/λ]
        # Only sum over visited bins where λ(x) > 0
        active = visited & (ratemap > 0)
        ratio = ratemap[active] / lambda_mean
        info_rate = (p_x[active] * ratemap[active] * np.log2(ratio)).sum()

        info_per_spike = info_rate / lambda_mean

        return {
            "information_rate": float(info_rate),
            "information_per_spike": float(info_per_spike),
            "mean_rate": float(lambda_mean),
            "ratemap": ratemap,
            "occupancy_prob": p_x,
        }

    def build_occupancy_map(
        self,
        positions: np.ndarray,
        arena_limits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a 2-D occupancy map from a sequence of (x, y) positions.

        Parameters
        ----------
        positions : ndarray, shape (T, 2)
            Each row is ``[x, y]`` position at time-step t.
        arena_limits : ndarray, shape (2, 2)
            ``[[x_min, x_max], [y_min, y_max]]``.

        Returns
        -------
        occupancy : ndarray, shape (n_bins_y, n_bins_x)
        x_edges   : ndarray, shape (n_bins_x + 1,)
        y_edges   : ndarray, shape (n_bins_y + 1,)
        """
        positions = np.asarray(positions)
        x = positions[:, 0]
        y = positions[:, 1]

        x_edges = np.linspace(arena_limits[0, 0], arena_limits[0, 1], self.n_bins_x + 1)
        y_edges = np.linspace(arena_limits[1, 0], arena_limits[1, 1], self.n_bins_y + 1)

        occupancy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        # Transpose so shape is (n_bins_y, n_bins_x) – consistent with imshow
        occupancy = occupancy.T
        return occupancy, x_edges, y_edges

    def build_ratemap(
        self,
        positions: np.ndarray,
        activations: np.ndarray,
        arena_limits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a mean-activation ratemap from raw position and activation data.

        Parameters
        ----------
        positions : ndarray, shape (T, 2)
            (x, y) positions at each time step.
        activations : ndarray, shape (T,)
            Scalar activation of a single unit at each time step.
        arena_limits : ndarray, shape (2, 2)

        Returns
        -------
        ratemap   : ndarray, shape (n_bins_y, n_bins_x)
        occupancy : ndarray, shape (n_bins_y, n_bins_x)
        """
        positions = np.asarray(positions)
        activations = np.asarray(activations, dtype=float)

        x = positions[:, 0]
        y = positions[:, 1]

        x_edges = np.linspace(arena_limits[0, 0], arena_limits[0, 1], self.n_bins_x + 1)
        y_edges = np.linspace(arena_limits[1, 0], arena_limits[1, 1], self.n_bins_y + 1)

        # Spike (activation) count per bin
        spike_count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=activations)
        occupancy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

        spike_count = spike_count.T
        occupancy = occupancy.T

        # Mean activation per bin
        with np.errstate(invalid="ignore", divide="ignore"):
            ratemap = np.where(occupancy > 0, spike_count / occupancy, 0.0)

        return ratemap, occupancy

    def compute_from_agent_env(
        self,
        agent,
        env,
        arena_limits: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """Compute information rate using ``agent.get_ratemap_matrix()`` and
        position data from ``env.history``.

        Works for any NeuralPlayground agent that:
          1. Overrides ``get_ratemap_matrix()`` to return a 2-D ndarray of
             shape ``(n_bins_y, n_bins_x)``, and
          2. Records position in ``env.history`` as ``state[0]`` and
             ``state[1]`` (x and y, standard NP convention).

        Agents that return ``None`` from ``get_ratemap_matrix()`` (e.g.
        ``RandomAgent``) are not compatible with this metric.  A warning is
        issued and ``None`` is returned.

        Parameters
        ----------
        agent : AgentCore subclass
        env   : Environment subclass
        arena_limits : ndarray, shape (2, 2), optional
            If not provided, inferred from ``env.arena_limits``.

        Returns
        -------
        dict or None
            Same keys as ``compute_from_ratemap``, plus ``positions``.
        """
        ratemap = agent.get_ratemap_matrix()
        if ratemap is None:
            warnings.warn(
                f"Agent '{type(agent).__name__}' returned None from "
                "get_ratemap_matrix().  This agent does not expose a spatial "
                "firing-rate map, so InformationRateScorer cannot be applied. "
                "For CSCG agents use compute_from_cscg_agent() instead.",
                UserWarning,
            )
            return None

        ratemap = np.asarray(ratemap, dtype=float)

        # Extract (x, y) positions from env.history
        history = env.history
        if len(history) == 0:
            warnings.warn("env.history is empty – no position data.", UserWarning)
            return None

        positions = np.array([[s["state"][0], s["state"][1]] for s in history])

        if arena_limits is None:
            if hasattr(env, "arena_limits"):
                arena_limits = np.asarray(env.arena_limits)
            else:
                arena_limits = np.array(
                    [[positions[:, 0].min(), positions[:, 0].max()], [positions[:, 1].min(), positions[:, 1].max()]]
                )

        occupancy, _, _ = self.build_occupancy_map(positions, arena_limits)

        # Resize ratemap to match occupancy if needed
        if ratemap.shape != occupancy.shape:
            from scipy.ndimage import zoom

            zf = (occupancy.shape[0] / ratemap.shape[0], occupancy.shape[1] / ratemap.shape[1])
            ratemap = zoom(ratemap, zf, order=1)

        result = self.compute_from_ratemap(ratemap, occupancy)
        result["positions"] = positions
        return result

    def compute_from_cscg_agent(
        self,
        agent,
        env,
        smoothing_sigma: Optional[float] = None,
    ) -> Optional[dict]:
        """Compute per-clone information rates for a ``George2021`` (CSCG) agent.

        The belief-state (forward message) of clone *i* at time *t* is treated
        as the instantaneous "firing rate" of that clone.  The mean activation
        at each spatial bin gives λ_i(x), and the Skaggs formula is applied
        per clone.

        Parameters
        ----------
        agent : George2021
            A trained CSCG agent.  Must have ``pos_history`` populated and
            ``get_belief_state()`` returning shape ``(T, n_clones)``.
        env : DiscreteObjectEnvironment
            The environment used during training.
        smoothing_sigma : float, optional
            Override the scorer's default ``smoothing_sigma`` for this call.

        Returns
        -------
        dict with keys:
            ``per_clone``        – list of dicts (one per clone), each with
                                   ``information_rate``, ``information_per_spike``,
                                   ``mean_rate``, ``ratemap``, ``occupancy_prob``
            ``mean_info_rate``   – mean information rate across clones (bits/step)
            ``mean_info_per_spike`` – mean bits/activation across clones
            ``best_clone_idx``   – index of the clone with highest information rate
            ``occupancy_prob``   – shared occupancy probability map
            ``place_fields``     – ndarray, shape (n_clones, n_bins_y, n_bins_x)
            ``n_clones``         – total number of clones
        """
        sigma = smoothing_sigma if smoothing_sigma is not None else self.smoothing_sigma

        # ---- 1. Gather belief states ----------------------------------------
        beliefs = agent.get_belief_state()  # shape (T, n_clones)
        if beliefs is None or len(beliefs) == 0:
            warnings.warn("get_belief_state() returned empty array.", UserWarning)
            return None

        # ---- 2. Gather positions --------------------------------------------
        # pos_history entries are [flat_state_index, one_hot_vector, (x, y)]
        flat_positions = []
        for ep in agent.episode_history:
            flat_positions.extend(ep["pos"])
        if hasattr(agent, "current_episode") and len(agent.current_episode["pos"]) > 0:
            flat_positions.extend(agent.current_episode["pos"])

        if len(flat_positions) == 0:
            warnings.warn("No position data found in agent history.", UserWarning)
            return None

        # Align lengths (belief states and positions may differ by 1 due to
        # the act-before-step ordering in the training loop)
        T = min(len(beliefs), len(flat_positions))
        beliefs = beliefs[:T]
        flat_positions = flat_positions[:T]

        n_clones = beliefs.shape[1]

        # ---- 3. Build discrete grid from arena layout -----------------------
        if hasattr(env, "custom_layout") and env.custom_layout is not None:
            grid_h, grid_w = env.custom_layout.shape
        else:
            grid_h = env.resolution_d
            grid_w = env.resolution_w

        grid_width = env.resolution_w  # columns in the flat index

        # ---- 4. Build per-clone activity maps and shared occupancy ----------
        # activity_maps[i, r, c] = sum of belief_state[i] over all t at (r, c)
        activity_maps = np.zeros((n_clones, grid_h, grid_w))
        occupancy_map = np.zeros((grid_h, grid_w))

        for t, pos in enumerate(flat_positions):
            flat_idx = int(pos[0])  # flat grid index
            r = flat_idx // grid_width
            c = flat_idx % grid_width
            if 0 <= r < grid_h and 0 <= c < grid_w:
                activity_maps[:, r, c] += beliefs[t]
                occupancy_map[r, c] += 1

        # Mean activation per bin: place_fields[i] = λ_i(x)
        occ_safe = np.where(occupancy_map > 0, occupancy_map, 1e-8)
        place_fields = activity_maps / occ_safe[None, :, :]  # (n_clones, ny, nx)

        # Optional smoothing
        if sigma > 0:
            for i in range(n_clones):
                place_fields[i] = gaussian_filter(place_fields[i], sigma=sigma)

        # Shared occupancy probability p(x)
        total_occ = occupancy_map.sum()
        if total_occ == 0:
            warnings.warn("All occupancy counts are zero.", UserWarning)
            return None
        p_x = occupancy_map / total_occ

        # ---- 5. Compute information rate per clone --------------------------
        per_clone_results = []
        visited = occupancy_map >= self.min_occupancy

        for i in range(n_clones):
            lam_x = place_fields[i]  # λ_i(x)

            # λ_i = Σ_x p(x)·λ_i(x)
            lambda_mean = (p_x[visited] * lam_x[visited]).sum()

            if lambda_mean == 0:
                per_clone_results.append(
                    {
                        "information_rate": 0.0,
                        "information_per_spike": 0.0,
                        "mean_rate": 0.0,
                        "ratemap": lam_x,
                        "occupancy_prob": p_x,
                    }
                )
                continue

            active = visited & (lam_x > 0)
            ratio = lam_x[active] / lambda_mean
            info_rate = (p_x[active] * lam_x[active] * np.log2(ratio)).sum()
            info_per_spike = info_rate / lambda_mean

            per_clone_results.append(
                {
                    "information_rate": float(info_rate),
                    "information_per_spike": float(info_per_spike),
                    "mean_rate": float(lambda_mean),
                    "ratemap": lam_x,
                    "occupancy_prob": p_x,
                }
            )

        # ---- 6. Aggregate statistics ----------------------------------------
        all_ir = [r["information_rate"] for r in per_clone_results]
        all_ips = [r["information_per_spike"] for r in per_clone_results]

        return {
            "per_clone": per_clone_results,
            "mean_info_rate": float(np.mean(all_ir)),
            "mean_info_per_spike": float(np.mean(all_ips)),
            "best_clone_idx": int(np.argmax(all_ir)),
            "occupancy_prob": p_x,
            "place_fields": place_fields,
            "n_clones": n_clones,
        }

    def plot_information_rate(
        self,
        result: dict,
        ax: Optional[plt.Axes] = None,
        title: str = "Information Rate",
    ) -> plt.Axes:
        """Plot the ratemap with information rate annotated in the title.

        Parameters
        ----------
        result : dict
            Output of ``compute_from_ratemap`` or ``compute_from_agent_env``.
        ax : matplotlib Axes, optional
        title : str

        Returns
        -------
        ax : matplotlib Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        ratemap = result["ratemap"]
        ir = result["information_rate"]
        ips = result["information_per_spike"]

        im = ax.imshow(ratemap, origin="upper", cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Mean activation")
        ax.set_title(
            f"{title}\n{ir:.3f} bits/step  |  {ips:.3f} bits/activation",
            fontsize=9,
        )
        ax.set_xlabel("x bin")
        ax.set_ylabel("y bin")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_cscg_information_rates(
        self,
        cscg_result: dict,
        top_n: int = 10,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """Plot ratemaps of the top-N clones by information rate.

        Parameters
        ----------
        cscg_result : dict
            Output of ``compute_from_cscg_agent``.
        top_n : int
            Number of highest-information-rate clones to display.
        figsize : tuple, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        per_clone = cscg_result["per_clone"]
        n_clones = cscg_result["n_clones"]
        top_n = min(top_n, n_clones)

        ir_values = [r["information_rate"] for r in per_clone]
        top_indices = np.argsort(ir_values)[::-1][:top_n]

        n_cols = min(top_n, 5)
        n_rows = int(np.ceil(top_n / n_cols))
        if figsize is None:
            figsize = (n_cols * 2.5, n_rows * 2.5)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for k, clone_idx in enumerate(top_indices):
            result = per_clone[clone_idx]
            ir = result["information_rate"]
            ips = result["information_per_spike"]
            ax = axes[k]
            im = ax.imshow(result["ratemap"], origin="upper", cmap="viridis", aspect="auto")
            ax.set_title(f"Clone {clone_idx}\n{ir:.3f} b/step | {ips:.3f} b/act", fontsize=7)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        for k in range(top_n, len(axes)):
            axes[k].axis("off")

        mean_ir = cscg_result["mean_info_rate"]
        best_idx = cscg_result["best_clone_idx"]
        fig.suptitle(
            f"Top-{top_n} clones by information rate  (mean across all: {mean_ir:.3f} bits/step, best clone: {best_idx})",
            fontsize=10,
        )
        fig.tight_layout()
        return fig

    def plot_info_rate_histogram(
        self,
        cscg_result: dict,
        ax: Optional[plt.Axes] = None,
        bins: int = 20,
    ) -> plt.Axes:
        """Histogram of per-clone information rates.

        Parameters
        ----------
        cscg_result : dict
            Output of ``compute_from_cscg_agent``.
        ax : matplotlib Axes, optional
        bins : int

        Returns
        -------
        ax : matplotlib Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))

        ir_values = [r["information_rate"] for r in cscg_result["per_clone"]]
        ax.hist(ir_values, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(
            cscg_result["mean_info_rate"],
            color="tomato",
            lw=2,
            linestyle="--",
            label=f"mean = {cscg_result['mean_info_rate']:.3f}",
        )
        ax.set_xlabel("Information rate (bits / time-step)")
        ax.set_ylabel("Number of clones")
        ax.set_title("Distribution of clone information rates")
        ax.legend()
        return ax
