import json
from pathlib import Path


def test_final_v3_notebook_is_single_run_orchestration() -> None:
    path = Path("notebooks/final_v3_all_experiments.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"final_v3_cell_{index}", "exec")
    assert 'pipeline_version = "final_v3"' not in text  # configuration remains centralized in YAML
    assert "run_all(" in text
    assert "DEBUG_V3 = False" in text
    assert "RUN_V3_HOLDOUT = True" in text
    assert "RUN_V3_GENERATION = True" in text
    assert "RUN_V3_SEMANTIC_PROXY = True" in text
    assert "FORCE_REBUILD_ACTIVATIONS = True" in text
    assert "FORCE_RETRAIN_GAUSSIAN = True" in text
    assert "RUN_GENERATION = False" in text
    assert "https://github.com/leolazzz/t_lab_interp.git" in text
    assert "subprocess.run(['git', 'clone', '--depth', '1'" in text
    for section in range(29):
        assert f"{section}. " in text
    assert text.index("run_all(") < text.index("required_final_artifacts")
