import json
import subprocess
import sys

import pytest

from roadseg.cli import main
from roadseg.config import Config


def test_cli_help_and_demo_data(tmp_path):
    process = subprocess.run(
        [sys.executable, "-m", "roadseg.cli", "--help"], capture_output=True, text=True
    )
    assert process.returncode == 0 and "fuse-masks" in process.stdout
    output = tmp_path / "data"
    assert main(["demo-data", "--output", str(output)]) == 0
    assert (output / "test/masks/shape-0.png").is_file()
    assert main(["demo-data", "--output", str(output)]) == 2
    assert (output / "test/masks/shape-0.png").is_file()


def test_config_unknown_fields_and_string_classes_rejected(tmp_path):
    file = tmp_path / "config.json"
    file.write_text(json.dumps({"unknown_setting": 3}))
    with pytest.raises(ValueError):
        Config.load(file)
    with pytest.raises(ValueError):
        Config(classes="road")
