import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    classes: tuple[str, ...] = ("background", "road")
    height: int = 128
    width: int = 192
    backbone: str = "compact"
    decoder_channels: int = 32
    imagenet_weights: bool = False
    train_backbone: bool = True
    batch_size: int = 2
    epochs: int = 5
    learning_rate: float = 0.001
    seed: int = 42
    ignore_id: int = 255
    augment: bool = True

    def __post_init__(self):
        if not isinstance(self.classes, (list, tuple)):
            raise ValueError("classes must be an ordered list of names")
        object.__setattr__(self, "classes", tuple(self.classes))
        if (
            not 2 <= len(self.classes) <= 254
            or any(not isinstance(x, str) or not x.strip() for x in self.classes)
            or len(set(self.classes)) != len(self.classes)
        ):
            raise ValueError(
                "Use 2–254 unique class names, starting with background at ID 0"
            )
        limits = {
            "height": (32, 2048),
            "width": (32, 2048),
            "decoder_channels": (4, 512),
            "batch_size": (1, 64),
            "epochs": (1, 10000),
            "seed": (0, 2**31 - 1),
        }
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {key}: expected integer {low}–{high}")
        if self.height % 32 or self.width % 32:
            raise ValueError("Image height and width must be divisible by 32")
        if self.backbone not in {"compact", "vgg19"}:
            raise ValueError("backbone must be compact or vgg19")
        if self.imagenet_weights and self.backbone != "vgg19":
            raise ValueError("ImageNet weights require vgg19")
        for key in ("imagenet_weights", "train_backbone", "augment"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if (
            not isinstance(self.learning_rate, (int, float))
            or isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 1
        ):
            raise ValueError("learning_rate must be finite and in (0,1]")
        if type(self.ignore_id) is not int or self.ignore_id != 255:
            raise ValueError("This version reserves label 255 for ignored pixels")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Config must be a JSON object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
