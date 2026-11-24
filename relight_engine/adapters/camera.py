from __future__ import annotations
from dataclasses import dataclass
import math
from ..contracts import CameraModel

@dataclass(frozen=True)
class CameraCalibration:
    camera: CameraModel
    source: str
    horizontal_fov_degrees: float
    calibrated: bool
    note: str


def camera_from_horizontal_fov(width: int, height: int, horizontal_fov_degrees: float,
                               *, source: str = "user_supplied") -> CameraCalibration:
    """Create pinhole intrinsics from an explicitly supplied horizontal FOV.

    This is a calibration assumption unless it came from trusted camera metadata or a
    calibration target. It must not be represented as measured camera intrinsics otherwise.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not math.isfinite(horizontal_fov_degrees) or not 1.0 < horizontal_fov_degrees < 179.0:
        raise ValueError("Horizontal field of view must be between 1 and 179 degrees.")
    fx = 0.5 * width / math.tan(math.radians(horizontal_fov_degrees) * 0.5)
    fy = fx
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    calibrated = source in {"exif", "camera_profile", "checkerboard", "user_calibrated"}
    note = "Measured/calibrated camera intrinsics." if calibrated else "FOV-derived approximation; not a measured camera calibration."
    camera = CameraModel(float(fx), float(fy), float(cx), float(cy))
    camera.validate()
    return CameraCalibration(camera, source, float(horizontal_fov_degrees), calibrated, note)
