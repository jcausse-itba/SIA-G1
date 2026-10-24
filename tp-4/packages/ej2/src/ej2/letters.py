"""
Module to handle letters task for Hopfield networks.

Exposes a single entry-point `run_letters_task()` which is used to initiate the task.
"""

__all__ = ["run_letters_task"]


from typing import Dict, Any


def run_letters_task(cfg: Dict[str, Any]) -> None:
    print("Running letters task...")
