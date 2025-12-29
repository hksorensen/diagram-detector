"""Data models for diagram detection results."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class DiagramDetection:
    """Single diagram detection."""

    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str = "diagram"
    class_id: int = 0

    def __post_init__(self):
        """Validate bbox coordinates."""
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 coordinates, got {len(self.bbox)}")
        if not all(isinstance(x, (int, float)) for x in self.bbox):
            raise ValueError("bbox coordinates must be numeric")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    @property
    def width(self) -> float:
        """Get detection width."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Get detection height."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Get detection area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Get detection center point."""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "bbox": list(self.bbox),
            "confidence": float(self.confidence),
            "class": self.class_name,
            "width": float(self.width),
            "height": float(self.height),
            "area": float(self.area),
        }


@dataclass
class DetectionResult:
    """Detection results for one image/page."""

    filename: str
    page_number: Optional[int] = None  # For PDFs
    has_diagram: bool = False
    count: int = 0
    confidence: float = 0.0  # Max confidence
    detections: List[DiagramDetection] = field(default_factory=list)
    image: Optional[np.ndarray] = None  # Original image (optional)
    image_width: int = 0
    image_height: int = 0

    def __post_init__(self):
        """Calculate derived fields."""
        if self.detections:
            self.has_diagram = True
            self.count = len(self.detections)
            self.confidence = max(d.confidence for d in self.detections)
        else:
            self.has_diagram = False
            self.count = 0
            self.confidence = 0.0

    def get_crop(self, index: int = 0, padding: int = 10) -> Optional[np.ndarray]:
        """
        Get cropped diagram region.

        Args:
            index: Detection index (0-based)
            padding: Pixels to add around bbox

        Returns:
            Cropped image as numpy array, or None if no image available
        """
        if self.image is None:
            return None

        if not 0 <= index < len(self.detections):
            raise IndexError(f"Detection index {index} out of range (0-{len(self.detections)-1})")

        detection = self.detections[index]
        x1, y1, x2, y2 = detection.bbox

        # Add padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(self.image.shape[1], int(x2) + padding)
        y2 = min(self.image.shape[0], int(y2) + padding)

        return self.image[y1:y2, x1:x2]

    def get_all_crops(self, padding: int = 10) -> List[np.ndarray]:
        """Get all diagram crops."""
        if self.image is None:
            return []
        return [self.get_crop(i, padding) for i in range(len(self.detections))]

    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON export.

        Args:
            include_image: Whether to include base64-encoded image
        """
        result = {
            "filename": self.filename,
            "has_diagram": self.has_diagram,
            "count": self.count,
            "confidence": float(self.confidence),
            "detections": [d.to_dict() for d in self.detections],
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

        if self.page_number is not None:
            result["page_number"] = self.page_number

        if include_image and self.image is not None:
            import base64
            from io import BytesIO

            # Convert to PIL Image and encode
            pil_img = Image.fromarray(self.image)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            result["image_base64"] = img_str

        return result

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to CSV row (simplified format)."""
        row = {
            "filename": self.filename,
            "has_diagram": self.has_diagram,
            "count": self.count,
            "max_confidence": float(self.confidence),
        }

        if self.page_number is not None:
            row["page_number"] = self.page_number

        return row

    def save_visualization(
        self, output_path: Path, line_width: int = 3, font_size: int = 20
    ) -> None:
        """
        Save image with bounding boxes drawn.

        Args:
            output_path: Where to save visualization
            line_width: Thickness of bbox lines
            font_size: Size of confidence labels
        """
        if self.image is None:
            raise ValueError("No image available for visualization")

        import cv2

        # Copy image
        vis_img = self.image.copy()

        # Draw each detection
        for detection in self.detections:
            x1, y1, x2, y2 = [int(x) for x in detection.bbox]

            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), line_width)  # Green

            # Draw confidence label
            label = f"{detection.confidence:.2f}"
            cv2.putText(
                vis_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size / 20.0,
                (0, 255, 0),
                2,
            )

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
