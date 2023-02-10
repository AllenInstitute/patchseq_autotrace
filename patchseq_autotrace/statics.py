from importlib.resources import files

MODEL_NAME_PATHS = {
    "Aspiny1.0": files('patchseq_autotrace') / "data/aspiny1.0.ckpt",
    "Spiny1.0": files('patchseq_autotrace') / "data/spiny1.0.ckpt",
}

INTENSITY_THRESHOLDS = {
    "Aspiny1.0": 50,
    "Spiny1.0": 252,
}
