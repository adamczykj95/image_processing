"""Subprocess entrypoint for a single training run.

Used by the sweep runner so each grid point gets its own process — avoiding CUDA
memory/state accumulating across repeated engine.fit() calls in one long-lived process,
and isolating any one grid point's crash from the rest of the sweep.

Protocol: the caller (jobs/sweep_job.py) writes runs/<run_id>/config.json (a RunConfig
dump) *before* launching this; this entrypoint just reads it back, reconstructs the
RunConfig, and calls run_training() — which harmlessly rewrites the same file.

Usage: python -m image_processing.jobs.train_job --project-root <path> --run-id <id>
"""
import argparse
import json
import sys
from pathlib import Path

from image_processing.core.config_schema import RunConfig
from image_processing.core.project import Project
from image_processing.ml.train import run_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    project = Project.open(Path(args.project_root))
    config_path = project.runs_dir / args.run_id / "config.json"
    run_config = RunConfig.model_validate(json.loads(config_path.read_text()))
    run_training(project, args.run_id, run_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
