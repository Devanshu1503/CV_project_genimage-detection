"""The 'install only what is missing' logic never selects packages that are already importable."""
from src.colab_setup import _requirement_names, missing_packages


def test_requirement_names_and_missing(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("# comment\nnumpy>=1.24\npyyaml\nscikit-learn  # inline\n"
                   "pillow[extra]>=10\ndefinitely-not-a-real-pkg-xyz==1.0\n", encoding="utf-8")
    assert _requirement_names(req) == ["numpy", "pyyaml", "scikit-learn", "pillow",
                                       "definitely-not-a-real-pkg-xyz"]
    # numpy/yaml/sklearn/PIL are installed here, so only the fake package is "missing".
    assert missing_packages(req) == ["definitely-not-a-real-pkg-xyz"]


def test_colab_requirements_exclude_preinstalled():
    from src.config import REPO_ROOT
    names = {n.lower() for n in _requirement_names(REPO_ROOT / "requirements-colab.txt")}
    assert not names & {"torch", "torchvision", "numpy", "pandas", "scikit-learn", "matplotlib",
                        "pillow", "scipy", "seaborn"}
