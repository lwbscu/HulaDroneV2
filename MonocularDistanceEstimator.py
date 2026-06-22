import os

import cv2
import numpy as np


class MonocularDistanceEstimator:
    """Estimate distance to a red circular target with known diameter."""

    def __init__(
        self,
        calibration_file_path="camera_calibration.npz",
        target_diameter_cm=10.0,
        min_contour_area=120,
    ):
        self.calibration_file_path = calibration_file_path
        self.target_diameter_cm = float(target_diameter_cm)
        self.min_contour_area = int(min_contour_area)
        self.warning_distance_cm = 80.0
        self.critical_distance_cm = 40.0
        self.laser_aim_offset_x_px = 0.0
        self.laser_aim_offset_y_px = 0.0
        self.camera_matrix, self.dist_coeffs = self.load_calibration_data()
        self.current_distance_info = {
            "detected": False,
            "distance_cm": None,
            "message": "not initialized",
        }

    def load_calibration_data(self):
        if not os.path.exists(self.calibration_file_path):
            print("错误：标定文件 '{}' 未找到。".format(self.calibration_file_path))
            return None, None

        try:
            with np.load(self.calibration_file_path) as data:
                if "camera_matrix" in data and "dist_coeffs" in data:
                    return data["camera_matrix"], data["dist_coeffs"]
                if "mtx" in data and "dist" in data:
                    return data["mtx"], data["dist"]
                print("错误：标定文件中未找到相机内参。")
                return None, None
        except Exception as e:
            print("错误：加载标定文件失败: {}".format(e))
            return None, None

    def set_reference_size(self, target_diameter_cm, _unused_height_cm=None):
        diameter = float(target_diameter_cm)
        if diameter <= 0:
            raise ValueError("target_diameter_cm must be positive")
        self.target_diameter_cm = diameter

    def set_laser_aim_offset(self, offset_x_px, offset_y_px):
        self.laser_aim_offset_x_px = float(offset_x_px)
        self.laser_aim_offset_y_px = float(offset_y_px)

    def _find_contours(self, mask):
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(result) == 2:
            contours, _ = result
        else:
            _, contours, _ = result
        return contours

    def _estimate_distance_cm(self, diameter_px):
        if self.camera_matrix is None:
            return None

        focal_length_px = (
            float(self.camera_matrix[0, 0]) + float(self.camera_matrix[1, 1])
        ) / 2.0
        if diameter_px <= 0 or self.target_diameter_cm <= 0 or focal_length_px <= 0:
            return None
        return float((self.target_diameter_cm * focal_length_px) / float(diameter_px))

    def _risk_level(self, distance_cm):
        if distance_cm <= self.critical_distance_cm:
            return "DANGER", (0, 0, 255)
        if distance_cm <= self.warning_distance_cm:
            return "CAUTION", (0, 165, 255)
        return "CLEAR", (0, 220, 0)

    def _draw_panel(self, frame, lines, color=(0, 255, 255)):
        if frame is None:
            return

        panel_height = 30 + 26 * max(1, len(lines))
        cv2.rectangle(frame, (8, 8), (560, panel_height), (8, 18, 28), -1)
        cv2.rectangle(frame, (8, 8), (560, panel_height), color, 2)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (18, 34 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                color,
                2,
                cv2.LINE_AA,
            )

    def _build_red_mask(self, original_image):
        # pyhula frames are displayed directly by matplotlib, so treat them as RGB.
        hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
        lower_red_1 = np.array([0, 55, 50], dtype=np.uint8)
        upper_red_1 = np.array([18, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([160, 55, 50], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask_1, mask_2)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _red_coverage_near_point(self, mask, center_x, center_y, radius_px=14):
        height, width = mask.shape[:2]
        x0 = max(0, int(round(center_x - radius_px)))
        x1 = min(width, int(round(center_x + radius_px + 1)))
        y0 = max(0, int(round(center_y - radius_px)))
        y1 = min(height, int(round(center_y + radius_px + 1)))
        if x0 >= x1 or y0 >= y1:
            return 0.0

        roi = mask[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0
        return float(cv2.countNonZero(roi)) / float(roi.size)

    def _laser_alignment_info(self, mask, target_x, target_y, target_radius_px, aim_x, aim_y):
        target_radius_px = max(1.0, float(target_radius_px))
        aim_distance_px = float(np.hypot(target_x - aim_x, target_y - aim_y))
        edge_guard_px = max(4.0, min(18.0, target_radius_px * 0.25))
        safe_radius_px = max(3.0, target_radius_px - edge_guard_px)
        sample_radius_px = max(4.0, min(12.0, target_radius_px * 0.14))
        red_coverage = self._red_coverage_near_point(
            mask,
            aim_x,
            aim_y,
            sample_radius_px,
        )
        hit_margin_px = safe_radius_px - aim_distance_px
        laser_spot_on_target = (
            target_radius_px >= 10.0
            and hit_margin_px >= 0.0
            and red_coverage >= 0.35
        )
        return {
            "laser_spot_on_target": bool(laser_spot_on_target),
            "laser_red_coverage": float(red_coverage),
            "laser_to_center_px": float(aim_distance_px),
            "laser_hit_margin_px": float(hit_margin_px),
            "laser_safe_radius_px": float(safe_radius_px),
            "laser_sample_radius_px": float(sample_radius_px),
        }

    def get_distance_frame(self, original_image, display_frame=None):
        if original_image is None:
            return display_frame

        if display_frame is None:
            display_frame = original_image.copy()

        if self.camera_matrix is None:
            self.current_distance_info = {
                "detected": False,
                "distance_cm": None,
                "message": "camera calibration missing",
            }
            self._draw_panel(
                display_frame,
                ["Red Circle Distance: calibration missing"],
                (0, 0, 255),
            )
            return display_frame

        frame_height, frame_width = original_image.shape[:2]
        mask = self._build_red_mask(original_image)
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        center_red_coverage = self._red_coverage_near_point(
            mask,
            frame_center_x,
            frame_center_y,
        )
        center_red_detected = center_red_coverage >= 0.20
        contours = self._find_contours(mask)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.62:
                continue

            (center_x, center_y), radius_px = cv2.minEnclosingCircle(contour)
            diameter_px = radius_px * 2.0
            if diameter_px < 8:
                continue

            circle_area = np.pi * radius_px * radius_px
            fill_ratio = area / max(1.0, circle_area)
            if fill_ratio < 0.45:
                continue

            distance_cm = self._estimate_distance_cm(diameter_px)
            if distance_cm is None:
                continue

            candidates.append(
                {
                    "center": (float(center_x), float(center_y)),
                    "radius_px": float(radius_px),
                    "diameter_px": float(diameter_px),
                    "area": float(area),
                    "circularity": float(circularity),
                    "fill_ratio": float(fill_ratio),
                    "distance_cm": distance_cm,
                    "center_offset": (
                        abs(center_x - frame_center_x) / max(1.0, frame_center_x)
                        + abs(center_y - frame_center_y) / max(1.0, frame_center_y)
                    )
                    / 2.0,
                }
            )

        candidates.sort(key=lambda item: (item["distance_cm"], item["center_offset"]))

        if not candidates:
            self.current_distance_info = {
                "detected": False,
                "distance_cm": None,
                "message": "red circle not detected",
                "target_diameter_cm": self.target_diameter_cm,
                "frame_size": (int(frame_width), int(frame_height)),
                "frame_center": (float(frame_center_x), float(frame_center_y)),
                "center_red_detected": bool(center_red_detected),
                "center_red_coverage": float(center_red_coverage),
                "laser_spot_on_target": False,
            }
            self._draw_panel(
                display_frame,
                [
                    "Red Circle Distance: no target",
                    "Waiting for red circle target",
                ],
                (0, 165, 255),
            )
            cv2.drawMarker(
                display_frame,
                (int(round(frame_center_x)), int(round(frame_center_y))),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=32,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            aim_x = frame_center_x + self.laser_aim_offset_x_px
            aim_y = frame_center_y + self.laser_aim_offset_y_px
            cv2.drawMarker(
                display_frame,
                (int(round(aim_x)), int(round(aim_y))),
                (255, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=30,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            return display_frame

        selected = candidates[0]
        risk_text, risk_color = self._risk_level(selected["distance_cm"])
        selected_x, selected_y = selected["center"]
        aim_x = frame_center_x + self.laser_aim_offset_x_px
        aim_y = frame_center_y + self.laser_aim_offset_y_px
        offset_x_px = selected_x - aim_x
        offset_y_px = selected_y - aim_y
        offset_x_ratio = offset_x_px / max(1.0, frame_center_x)
        offset_y_ratio = offset_y_px / max(1.0, frame_center_y)
        laser_alignment = self._laser_alignment_info(
            mask,
            selected_x,
            selected_y,
            selected["radius_px"],
            aim_x,
            aim_y,
        )

        for index, candidate in enumerate(candidates[:3]):
            center_x, center_y = candidate["center"]
            radius_px = int(round(candidate["radius_px"]))
            color = risk_color if index == 0 else (180, 180, 180)
            thickness = 3 if index == 0 else 1
            cv2.circle(
                display_frame,
                (int(round(center_x)), int(round(center_y))),
                radius_px,
                color,
                thickness,
            )
            cv2.putText(
                display_frame,
                "{:.1f}cm".format(candidate["distance_cm"]),
                (
                    int(round(center_x - radius_px)),
                    max(24, int(round(center_y - radius_px - 8))),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        selected_radius = int(round(selected["radius_px"]))
        self.current_distance_info = {
            "detected": True,
            "distance_cm": float(selected["distance_cm"]),
            "risk": risk_text,
            "frame_size": (int(frame_width), int(frame_height)),
            "center": selected["center"],
            "frame_center": (float(frame_center_x), float(frame_center_y)),
            "center_red_detected": bool(center_red_detected),
            "center_red_coverage": float(center_red_coverage),
            "aim_point": (float(aim_x), float(aim_y)),
            "laser_aim_offset_x_px": float(self.laser_aim_offset_x_px),
            "laser_aim_offset_y_px": float(self.laser_aim_offset_y_px),
            "offset_x_px": float(offset_x_px),
            "offset_y_px": float(offset_y_px),
            "offset_x_ratio": float(offset_x_ratio),
            "offset_y_ratio": float(offset_y_ratio),
            "radius_px": selected["radius_px"],
            "diameter_px": selected["diameter_px"],
            "area": selected["area"],
            "circularity": selected["circularity"],
            "fill_ratio": selected["fill_ratio"],
            "target_diameter_cm": self.target_diameter_cm,
        }
        self.current_distance_info.update(laser_alignment)

        laser_on_paper = laser_alignment["laser_spot_on_target"]
        laser_status = "ON PAPER" if laser_on_paper else "OFF PAPER"

        self._draw_panel(
            display_frame,
            [
                "Red Circle Distance: {:.1f} cm | {}".format(
                    selected["distance_cm"], risk_text
                ),
                "Diameter: {:.1f}px | Real: {:.1f}cm | Round: {:.2f}".format(
                    selected["diameter_px"],
                    self.target_diameter_cm,
                    selected["circularity"],
                ),
                "Offset: X {:+.0f}px Y {:+.0f}px".format(offset_x_px, offset_y_px),
                "Laser spot: {} | red {:.0%} | margin {:+.0f}px".format(
                    laser_status,
                    laser_alignment["laser_red_coverage"],
                    laser_alignment["laser_hit_margin_px"],
                ),
            ],
            risk_color,
        )
        center_point = (int(round(frame_center_x)), int(round(frame_center_y)))
        cv2.drawMarker(
            display_frame,
            center_point,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        aim_point = (int(round(aim_x)), int(round(aim_y)))
        cv2.drawMarker(
            display_frame,
            aim_point,
            (255, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=30,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        safe_radius = int(round(laser_alignment["laser_safe_radius_px"]))
        cv2.circle(
            display_frame,
            (int(round(selected_x)), int(round(selected_y))),
            safe_radius,
            (0, 255, 0) if laser_on_paper else (255, 0, 255),
            1,
        )
        cv2.circle(
            display_frame,
            aim_point,
            6,
            (0, 255, 0) if laser_on_paper else (255, 0, 255),
            -1 if laser_on_paper else 2,
        )
        cv2.line(
            display_frame,
            aim_point,
            (int(round(selected_x)), int(round(selected_y))),
            (0, 255, 0) if laser_on_paper else (255, 0, 255),
            1,
        )
        cv2.circle(
            display_frame,
            (int(round(selected_x)), int(round(selected_y))),
            5,
            risk_color,
            -1,
        )
        cv2.line(
            display_frame,
            (int(round(selected_x - selected_radius)), int(round(selected_y))),
            (int(round(selected_x + selected_radius)), int(round(selected_y))),
            risk_color,
            2,
        )
        return display_frame
