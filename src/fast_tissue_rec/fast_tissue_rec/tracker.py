from collections.abc import Sequence
import cv2 as cv
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class KLTTracker:
    """
    Kanade-Lucas-Tomasi feature tracking

    Uses AKAZE feature detector to generate candidate features, then uses KLT to track those features forward.
    Once enough features leave the field of view, or are lost track of, new features are sampled to replace them.
    Care is taken to prevent clustering and ensure good spatial diversity of features, as well as avoiding tracking specularities.
    """

    class Feature:
        def __init__(self, id: int, point: npt.NDArray[np.float32]):
            self.id = id
            self.point = point

    def __init__(
        self,
        image_shape: tuple[int, int],
        track_count_bounds: tuple[int, int] = (240, 300),
        anti_cluster_grid: tuple[int, int] = (12, 12),
        max_tracks_per_cell: int = 4,
        apply_clahe_normalization: bool = True,
    ):
        self.image_shape = image_shape
        self.anti_cluster_grid = anti_cluster_grid

        self.track_count_bounds = track_count_bounds
        self.max_tracks_per_cell = max_tracks_per_cell

        image_height, image_width = self.image_shape[:2]
        cols, rows = self.anti_cluster_grid
        self.cell_width = image_width / cols
        self.cell_height = image_height / rows

        self.akaze_detector = cv.AKAZE_create()
        self.forward_backward_consistency_threshold = 3.0

        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.use_clahe = apply_clahe_normalization

        self.lk_parameters = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.next_feature_id = 0  # monotonically-increasing feature ID

        self.feature_tracks: list["KLTTracker.Feature"] = []

    def _grid_coordinates(self, point: npt.NDArray) -> tuple[int, int]:
        """Convert pixel coordinates to anti-cluster grid (column, row) indices"""
        col_idx = min(int(point[0] // self.cell_width), self.anti_cluster_grid[0] - 1)
        row_idx = min(int(point[1] // self.cell_height), self.anti_cluster_grid[1] - 1)
        return (col_idx, row_idx)

    def _get_grid_occupancy(
        self, current_feature_tracks: Sequence["KLTTracker.Feature"]
    ) -> npt.NDArray[np.int32]:
        """Returns number of active tracks per each anti-cluster grid cell"""
        cols, rows = self.anti_cluster_grid
        grid_counts = np.zeros((rows, cols), dtype=np.int32)

        for feature in current_feature_tracks:
            grid_counts[*self._grid_coordinates(feature.point)] += 1

        return grid_counts

    def _specularity_mask(
        self, image: npt.NDArray, threshold: int = 230, dilate_radius: int = 50
    ) -> npt.NDArray:
        """Provides positive mask of likely specularities. `image` may be color or grayscale."""
        if threshold < 0 or threshold > 255:
            raise ValueError(f"threshold must be in [0,255], got {threshold}")
        if dilate_radius < 0:
            raise ValueError(f"dilate_radius must be >= 0, got {dilate_radius}")

        if image.ndim == 3 and image.shape[2] > 1:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        _, mask = cv.threshold(image, int(threshold), 255, cv.THRESH_BINARY)
        if dilate_radius == 0:
            return mask

        r = int(dilate_radius)
        k = 2 * r + 1

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        mask = cv.dilate(mask, kernel, iterations=1)
        return mask

    def _sample_new_features(
        self, image: npt.NDArray, current_tracks: Sequence["KLTTracker.Feature"]
    ) -> list["KLTTracker.Feature"]:
        """Detects new AKAZE features and sample new tracks, avoiding forming tight clusters"""
        grid_counts = self._get_grid_occupancy(current_tracks)

        candidate_features = self.akaze_detector.detect(image, None)
        if not candidate_features:
            return list(current_tracks)

        # Create a binary mask to prevent picking points too close to existing points
        exclusion_radius = 20
        exclusion_mask = np.ones(image.shape, dtype=np.uint8) * 255
        for track in current_tracks:
            x, y = map(int, track.point)
            cv.circle(exclusion_mask, (x, y), exclusion_radius, 0, cv.FILLED)

        non_specularities = 255 - self._specularity_mask(image)
        exclusion_mask = cv.bitwise_or(exclusion_mask, non_specularities)

        new_tracks = list(current_tracks)

        # sort highest to lowest resposne so we start with the best features
        candidate_features = sorted(
            candidate_features, key=lambda x: x.response, reverse=True
        )
        for keypoint in candidate_features:
            if len(new_tracks) >= self.track_count_bounds[1]:
                break

            col_idx, row_idx = self._grid_coordinates(keypoint.pt)

            # Skip this keypoint if the local region is already well-represented
            if grid_counts[row_idx, col_idx] >= self.max_tracks_per_cell:
                continue

            minimum_patch = 6.0
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])

            # Ensure we don't sample points too close to image boundary, since that makes optical
            # flow hard and also more likely that point will go out of view very quickly
            if (
                minimum_patch <= x < self.image_shape[1] - minimum_patch
                and minimum_patch <= y < self.image_shape[0] - minimum_patch
            ):
                # Also make sure we don't track specularities/existing features/etc
                if exclusion_mask[y, x] < 255:
                    continue

                new_tracks.append(
                    KLTTracker.Feature(
                        self.next_feature_id, np.array(keypoint.pt, dtype=np.float32)
                    )
                )
                self.next_feature_id += 1
                grid_counts[row_idx, col_idx] += 1

        return new_tracks

    def _tracks_to_features(
        self, tracks: list["KLTTracker.Feature"]
    ) -> list[npt.NDArray[np.float32]]:
        return np.array([f.point for f in tracks], dtype=np.float32).reshape(-1, 1, 2)

    def initialize(self, frame: npt.NDArray[np.uint8]):
        raw_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.previous_frame = self.clahe.apply(raw_gray) if self.use_clahe else raw_gray
        self.feature_tracks = self._sample_new_features(self.previous_frame, [])

    def track(
        self, frame: npt.NDArray[np.uint8], verbose=False
    ) -> list["KLTTracker.Feature"]:
        raw_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = self.clahe.apply(raw_gray) if self.use_clahe else raw_gray
        previous_points = self._tracks_to_features(self.feature_tracks)

        # Forward/backward consistency check - drop ambiguous tracks if following flow backwards
        # doesn't lead back to where we came from
        next_points, status_forward, _ = cv.calcOpticalFlowPyrLK(
            self.previous_frame,
            gray,
            previous_points,
            None,
            **self.lk_parameters,
        )
        retros, status_backward, _ = cv.calcOpticalFlowPyrLK(
            gray, self.previous_frame, next_points, None, **self.lk_parameters
        )

        status_forward = status_forward.flatten()
        status_backward = status_backward.flatten()

        updated_tracks = []
        for idx, track in enumerate(self.feature_tracks):
            # Drop track if tracking failed outright
            if status_forward[idx] != 1 or status_backward[idx] != 1:
                continue

            previous = previous_points[idx].ravel()
            next = next_points[idx].ravel()
            retro = retros[idx].ravel()

            # or if forward/backward tracking is not self-consistent
            error = np.linalg.norm(previous - retro)
            if error > self.forward_backward_consistency_threshold:
                continue

            # calcOpticalFlowPyrLK can predict points outside the image
            if (
                0 <= next[0] < self.image_shape[1]
                and 0 <= next[1] < self.image_shape[0]
            ):
                updated_tracks.append(KLTTracker.Feature(track.id, next))

        self.feature_tracks = updated_tracks
        if len(self.feature_tracks) < self.track_count_bounds[0]:
            n = len(self.feature_tracks)
            self.feature_tracks = self._sample_new_features(gray, self.feature_tracks)
            if verbose:
                print(
                    f"\nHad only {n} features, sampled up to total of {len(self.feature_tracks)}"
                )

        self.previous_frame = gray.copy()

        return self.feature_tracks


def track(
    frames: list[npt.NDArray[np.uint8]], output_subsample: int
) -> list[npt.NDArray[np.float32]]:
    tracker = KLTTracker(frames[0].shape)
    tracker.initialize(frames[0])

    feature_observations = []

    for idx, frame in enumerate(tqdm(frames)):
        tracks = tracker.track(frame)

        if idx % output_subsample != 0:
            continue

        frame = np.array(
            [
                [track.id, track.point[0].item(), track.point[1].item()]
                for track in tracks
            ],
            dtype=np.float32,
        )
        feature_observations.append(frame)

    return feature_observations


__all__ = ["KLTTracker", "track"]
