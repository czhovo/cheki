from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import ConvexHull


def order_quad(points):
    points = np.asarray(points, dtype=np.float64)
    sums = points[:, 0] + points[:, 1]
    differences = points[:, 0] - points[:, 1]
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmax(differences)],
            points[np.argmax(sums)],
            points[np.argmin(differences)],
        ],
        dtype=np.float64,
    )


def points_inside_quad(points, quad):
    points = np.asarray(points, dtype=np.float64)
    quad = np.asarray(quad, dtype=np.float64)
    signed_area = sum(
        quad[index, 0] * quad[(index + 1) % 4, 1]
        - quad[(index + 1) % 4, 0] * quad[index, 1]
        for index in range(4)
    )
    clockwise = signed_area > 0
    inside = np.ones(len(points), dtype=bool)
    for index in range(4):
        first, second = quad[index], quad[(index + 1) % 4]
        dx, dy = second[0] - first[0], second[1] - first[1]
        cross = (
            dx * (points[:, 1] - first[1])
            - dy * (points[:, 0] - first[0])
        )
        inside &= (cross >= 0) if clockwise else (cross <= 0)
    return inside


def shrink_quad(vertices, factor=0.03):
    vertices = np.asarray(vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    return vertices + factor * (center - vertices)


def line_from_points(first, second):
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    if abs(dx) > abs(dy):
        slope = dy / (dx + 1e-9)
        return "h", slope, first[1] - slope * first[0]
    slope = dx / (dy + 1e-9)
    return "v", slope, first[0] - slope * first[1]


def ransac_quad_fit(exterior, initial_quad, filter_dist=80, inlier_dist=5):
    initial_lines = [
        line_from_points(initial_quad[0], initial_quad[1]),
        line_from_points(initial_quad[1], initial_quad[2]),
        line_from_points(initial_quad[2], initial_quad[3]),
        line_from_points(initial_quad[3], initial_quad[0]),
    ]
    exterior_x, exterior_y = exterior[:, 0], exterior[:, 1]
    distances = []
    for orientation, slope, intercept in initial_lines:
        if orientation == "h":
            distances.append(
                np.abs(exterior_y - (slope * exterior_x + intercept))
            )
        else:
            distances.append(
                np.abs(exterior_x - (slope * exterior_y + intercept))
            )
    edge_indices = np.argmin(np.column_stack(distances), axis=1)

    def fit_line(points, orientation, initial_slope, initial_intercept):
        if len(points) < 5:
            return orientation, initial_slope, initial_intercept
        if orientation == "h":
            points = points[
                np.abs(
                    points[:, 1]
                    - (initial_slope * points[:, 0] + initial_intercept)
                )
                < filter_dist
            ]
            x = points[:, 0:1].astype(np.float32)
            y = points[:, 1:2].astype(np.float32)
        else:
            points = points[
                np.abs(
                    points[:, 0]
                    - (initial_slope * points[:, 1] + initial_intercept)
                )
                < filter_dist
            ]
            x = points[:, 1:2].astype(np.float32)
            y = points[:, 0:1].astype(np.float32)
        if len(points) < 5:
            return orientation, initial_slope, initial_intercept

        best_inliers = 0
        best_model = (initial_slope, initial_intercept)
        random = np.random.default_rng(42)
        for _ in range(min(500, len(points) * 10)):
            indices = random.choice(len(points), min(5, len(points)), replace=False)
            design = np.hstack([x[indices], np.ones((len(indices), 1))])
            try:
                coefficients = np.linalg.lstsq(
                    design, y[indices], rcond=None
                )[0]
            except np.linalg.LinAlgError:
                continue
            slope = float(coefficients[0, 0])
            intercept = float(coefficients[1, 0])
            inliers = int(
                np.sum(np.abs(y - (slope * x + intercept)) < inlier_dist)
            )
            if inliers > best_inliers:
                best_inliers = inliers
                best_model = (slope, intercept)

        slope, intercept = best_model
        inlier_mask = (
            np.abs(y - (slope * x + intercept)) < inlier_dist
        ).flatten()
        if inlier_mask.sum() >= 3:
            design = np.hstack(
                [x[inlier_mask], np.ones((inlier_mask.sum(), 1))]
            )
            coefficients = np.linalg.lstsq(
                design, y[inlier_mask], rcond=None
            )[0]
            return (
                orientation,
                float(coefficients[0, 0]),
                float(coefficients[1, 0]),
            )
        return orientation, best_model[0], best_model[1]

    refined = [
        fit_line(exterior[edge_indices == index], *initial_lines[index])
        for index in range(4)
    ]

    def standard_form(line):
        orientation, slope, intercept = line
        if orientation == "h":
            return np.array([-slope, 1.0, -intercept], dtype=np.float64)
        return np.array([1.0, -slope, -intercept], dtype=np.float64)

    def intersect(first, second):
        first_a, first_b, first_c = standard_form(first)
        second_a, second_b, second_c = standard_form(second)
        denominator = first_a * second_b - second_a * first_b
        if abs(denominator) < 1e-9:
            return None
        x = (first_b * second_c - second_b * first_c) / denominator
        y = (first_c * second_a - second_c * first_a) / denominator
        return np.array([x, y], dtype=np.float64)

    corners = [
        intersect(refined[0], refined[3]),
        intersect(refined[0], refined[1]),
        intersect(refined[2], refined[1]),
        intersect(refined[2], refined[3]),
    ]
    result = np.array(
        [
            corner if corner is not None else initial_quad[index]
            for index, corner in enumerate(corners)
        ],
        dtype=np.float64,
    )
    return order_quad(result)


def mask_to_quad_ransac(mask, filter_dist=80, inlier_dist=5):
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("empty contour")
    points = np.vstack([contour.reshape(-1, 2) for contour in contours])
    if len(points) < 8:
        raise ValueError("not enough contour points")

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    differences = hull_points[:, None, :] - hull_points[None, :, :]
    distances_squared = np.sum(differences**2, axis=2)
    first, second = np.unravel_index(
        np.argmax(distances_squared), distances_squared.shape
    )
    first_point, second_point = hull_points[first], hull_points[second]
    vector = second_point - first_point
    norm = np.linalg.norm(vector)
    if norm < 1:
        raise ValueError("degenerate hull")
    signed = (
        vector[0] * (hull_points[:, 1] - first_point[1])
        - vector[1] * (hull_points[:, 0] - first_point[0])
    ) / norm
    approximate = np.array(
        [
            first_point,
            second_point,
            hull_points[np.argmax(signed)],
            hull_points[np.argmin(signed)],
        ],
        dtype=np.float64,
    )
    angles = np.arctan2(
        approximate[:, 1] - approximate.mean(axis=0)[1],
        approximate[:, 0] - approximate.mean(axis=0)[0],
    )
    approximate = approximate[np.argsort(angles)]
    exterior = points[
        ~points_inside_quad(points, shrink_quad(approximate, 0.03))
    ]
    if len(exterior) < 4:
        exterior = points

    rectangle = cv2.minAreaRect(hull_points.astype(np.float32))
    initial_quad = cv2.boxPoints(rectangle)
    initial_angles = np.arctan2(
        initial_quad[:, 1] - initial_quad.mean(axis=0)[1],
        initial_quad[:, 0] - initial_quad.mean(axis=0)[0],
    )
    initial_quad = initial_quad[np.argsort(initial_angles)]
    return ransac_quad_fit(
        exterior,
        initial_quad,
        filter_dist=filter_dist,
        inlier_dist=inlier_dist,
    )


def _normalized_standard_form(line):
    orientation, slope, intercept = line
    if orientation == "h":
        coefficients = np.array([-slope, 1.0, -intercept], dtype=np.float64)
    else:
        coefficients = np.array([1.0, -slope, -intercept], dtype=np.float64)
    norm = float(np.linalg.norm(coefficients[:2]))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    coefficients /= norm
    if coefficients[0] < 0 or (
        abs(coefficients[0]) < 1e-12 and coefficients[1] < 0
    ):
        coefficients *= -1.0
    return coefficients


def ransac_quad_fit_diagnostics(
    exterior,
    initial_quad,
    filter_dist=80,
    inlier_dist=5,
):
    """Run the frozen fit and return read-only line diagnostics.

    This intentionally mirrors ``ransac_quad_fit`` instead of changing its code
    path. The audit's equivalence gate verifies that the returned quadrilateral
    is identical to the frozen implementation before full collection begins.
    """
    exterior = np.asarray(exterior)
    initial_quad = np.asarray(initial_quad)
    initial_lines = [
        line_from_points(initial_quad[0], initial_quad[1]),
        line_from_points(initial_quad[1], initial_quad[2]),
        line_from_points(initial_quad[2], initial_quad[3]),
        line_from_points(initial_quad[3], initial_quad[0]),
    ]
    exterior_x, exterior_y = exterior[:, 0], exterior[:, 1]
    distances = []
    for orientation, slope, intercept in initial_lines:
        if orientation == "h":
            distances.append(
                np.abs(exterior_y - (slope * exterior_x + intercept))
            )
        else:
            distances.append(
                np.abs(exterior_x - (slope * exterior_y + intercept))
            )
    edge_indices = np.argmin(np.column_stack(distances), axis=1)

    def fit_line(points, orientation, initial_slope, initial_intercept):
        assigned = np.asarray(points)
        if len(assigned) < 5:
            return (
                (orientation, initial_slope, initial_intercept),
                assigned,
                assigned,
            )
        if orientation == "h":
            filtered = assigned[
                np.abs(
                    assigned[:, 1]
                    - (initial_slope * assigned[:, 0] + initial_intercept)
                )
                < filter_dist
            ]
            x = filtered[:, 0:1].astype(np.float32)
            y = filtered[:, 1:2].astype(np.float32)
        else:
            filtered = assigned[
                np.abs(
                    assigned[:, 0]
                    - (initial_slope * assigned[:, 1] + initial_intercept)
                )
                < filter_dist
            ]
            x = filtered[:, 1:2].astype(np.float32)
            y = filtered[:, 0:1].astype(np.float32)
        if len(filtered) < 5:
            return (
                (orientation, initial_slope, initial_intercept),
                assigned,
                filtered,
            )

        best_inliers = 0
        best_model = (initial_slope, initial_intercept)
        random = np.random.default_rng(42)
        for _ in range(min(500, len(filtered) * 10)):
            indices = random.choice(
                len(filtered), min(5, len(filtered)), replace=False
            )
            design = np.hstack([x[indices], np.ones((len(indices), 1))])
            try:
                coefficients = np.linalg.lstsq(design, y[indices], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            slope = float(coefficients[0, 0])
            intercept = float(coefficients[1, 0])
            inliers = int(
                np.sum(np.abs(y - (slope * x + intercept)) < inlier_dist)
            )
            if inliers > best_inliers:
                best_inliers = inliers
                best_model = (slope, intercept)

        slope, intercept = best_model
        inlier_mask = (
            np.abs(y - (slope * x + intercept)) < inlier_dist
        ).flatten()
        if inlier_mask.sum() >= 3:
            design = np.hstack(
                [x[inlier_mask], np.ones((inlier_mask.sum(), 1))]
            )
            coefficients = np.linalg.lstsq(design, y[inlier_mask], rcond=None)[0]
            return (
                (
                    orientation,
                    float(coefficients[0, 0]),
                    float(coefficients[1, 0]),
                ),
                assigned,
                filtered,
            )
        return (
            (orientation, best_model[0], best_model[1]),
            assigned,
            filtered,
        )

    fitted = [
        fit_line(exterior[edge_indices == index], *initial_lines[index])
        for index in range(4)
    ]
    refined = [value[0] for value in fitted]

    # Use the exact frozen intersection arithmetic for the returned quad.
    def frozen_standard_form(line):
        orientation, slope, intercept = line
        if orientation == "h":
            return np.array([-slope, 1.0, -intercept], dtype=np.float64)
        return np.array([1.0, -slope, -intercept], dtype=np.float64)

    def frozen_intersect(first, second):
        first_a, first_b, first_c = frozen_standard_form(first)
        second_a, second_b, second_c = frozen_standard_form(second)
        denominator = first_a * second_b - second_a * first_b
        if abs(denominator) < 1e-9:
            return None
        x = (first_b * second_c - second_b * first_c) / denominator
        y = (first_c * second_a - second_c * first_a) / denominator
        return np.array([x, y], dtype=np.float64)

    corners = [
        frozen_intersect(refined[0], refined[3]),
        frozen_intersect(refined[0], refined[1]),
        frozen_intersect(refined[2], refined[1]),
        frozen_intersect(refined[2], refined[3]),
    ]
    raw_result = np.array(
        [
            corner if corner is not None else initial_quad[index]
            for index, corner in enumerate(corners)
        ],
        dtype=np.float64,
    )
    result = order_quad(raw_result)

    line_diagnostics = []
    for index, (line, assigned, filtered) in enumerate(fitted):
        coefficients = _normalized_standard_form(line)
        if len(filtered):
            residuals = np.abs(
                filtered[:, 0] * coefficients[0]
                + filtered[:, 1] * coefficients[1]
                + coefficients[2]
            )
            inlier_mask = residuals < float(inlier_dist)
            inlier_points = filtered[inlier_mask]
        else:
            residuals = np.empty(0, dtype=np.float64)
            inlier_mask = np.zeros(0, dtype=bool)
            inlier_points = filtered
        direction = np.array([-coefficients[1], coefficients[0]], np.float64)
        if len(inlier_points):
            projected = np.asarray(inlier_points, np.float64) @ direction
            support_span = float(projected.max() - projected.min())
        else:
            support_span = 0.0
        first = raw_result[index]
        second = raw_result[(index + 1) % 4]
        edge_length = float(np.linalg.norm(second - first))
        line_diagnostics.append(
            {
                "edge_index": index,
                "orientation": line[0],
                "assigned_support_count": int(len(assigned)),
                "filtered_support_count": int(len(filtered)),
                "inlier_count": int(inlier_mask.sum()),
                "inlier_ratio": float(inlier_mask.mean()) if len(filtered) else 0.0,
                "median_abs_normal_residual_px": float(np.median(residuals))
                if len(residuals)
                else None,
                "rms_normal_residual_px": float(np.sqrt(np.mean(residuals**2)))
                if len(residuals)
                else None,
                "p95_abs_normal_residual_px": float(np.percentile(residuals, 95))
                if len(residuals)
                else None,
                "support_span_px": support_span,
                "support_span_ratio_to_fitted_edge": support_span
                / max(edge_length, 1e-9),
                "fitted_edge_length_px": edge_length,
                "line_angle_rad": float(np.arctan2(direction[1], direction[0]) % np.pi),
                "line_offset_px": float(coefficients[2]),
                "line_a": float(coefficients[0]),
                "line_b": float(coefficients[1]),
            }
        )
    return result, {"lines": line_diagnostics}


def mask_to_quad_ransac_diagnostics(mask, filter_dist=80, inlier_dist=5):
    """Frozen mask-to-quad path plus read-only diagnostics for audit use."""
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("empty contour")
    points = np.vstack([contour.reshape(-1, 2) for contour in contours])
    if len(points) < 8:
        raise ValueError("not enough contour points")

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    differences = hull_points[:, None, :] - hull_points[None, :, :]
    distances_squared = np.sum(differences**2, axis=2)
    first, second = np.unravel_index(
        np.argmax(distances_squared), distances_squared.shape
    )
    first_point, second_point = hull_points[first], hull_points[second]
    vector = second_point - first_point
    norm = np.linalg.norm(vector)
    if norm < 1:
        raise ValueError("degenerate hull")
    signed = (
        vector[0] * (hull_points[:, 1] - first_point[1])
        - vector[1] * (hull_points[:, 0] - first_point[0])
    ) / norm
    approximate = np.array(
        [
            first_point,
            second_point,
            hull_points[np.argmax(signed)],
            hull_points[np.argmin(signed)],
        ],
        dtype=np.float64,
    )
    angles = np.arctan2(
        approximate[:, 1] - approximate.mean(axis=0)[1],
        approximate[:, 0] - approximate.mean(axis=0)[0],
    )
    approximate = approximate[np.argsort(angles)]
    exterior = points[
        ~points_inside_quad(points, shrink_quad(approximate, 0.03))
    ]
    if len(exterior) < 4:
        exterior = points

    rectangle = cv2.minAreaRect(hull_points.astype(np.float32))
    initial_quad = cv2.boxPoints(rectangle)
    initial_angles = np.arctan2(
        initial_quad[:, 1] - initial_quad.mean(axis=0)[1],
        initial_quad[:, 0] - initial_quad.mean(axis=0)[0],
    )
    initial_quad = initial_quad[np.argsort(initial_angles)]
    return ransac_quad_fit_diagnostics(
        exterior,
        initial_quad,
        filter_dist=filter_dist,
        inlier_dist=inlier_dist,
    )
