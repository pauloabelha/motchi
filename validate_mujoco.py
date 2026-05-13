"""Validate official MuJoCo physics stepping and GLFW viewer launch.

Run:
    python validate_mujoco.py
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from motchi.runtime.logging import error, info


MODEL_XML = """
<mujoco model="motchi_smoke_test">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.3 0.35 1"/>
    <body name="body" pos="0 0 1">
      <joint name="free" type="free"/>
      <geom type="sphere" size="0.15" rgba="0.9 0.45 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run(headless: bool, seconds: float) -> None:
    info("Creating MuJoCo smoke-test model")
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    for _ in range(240):
        mujoco.mj_step(model, data)
    info("Physics stepping OK")

    if headless:
        info("Headless validation complete")
        return

    info("Launching MuJoCo GLFW viewer")
    start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        info("Rendering active")
        while viewer.is_running() and time.monotonic() - start < seconds:
            step_start = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - step_start)))

    info("Viewer closed cleanly")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MuJoCo physics and viewer startup.")
    parser.add_argument("--headless", action="store_true", help="Skip viewer launch.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Viewer runtime before exit.")
    args = parser.parse_args()

    try:
        run(headless=args.headless, seconds=args.seconds)
    except Exception as exc:
        error(f"MuJoCo validation failed: {exc}")
        raise


if __name__ == "__main__":
    main()

