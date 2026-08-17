"""
Integration test for Streamlit app to verify Part A animation controls and Part B training execution.
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Resolve path to app.py at repo root regardless of where pytest is invoked from
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_PATH = str(_REPO_ROOT / "app.py")

# Ensure src/ is importable when running from the tests/ directory
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_streamlit_app_part_a_and_b():
    """Run simulated Streamlit test verifying Part A animation controls and Part B training."""
    at = AppTest.from_file(_APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit app raised an exception on load: {at.exception}"

    # Verify initial state of Part A
    assert at.session_state.part_a_iteration_slider == 0
    assert at.session_state.is_playing is False

    # 1. Step button interaction: advance by exactly 1 step
    at.button(key="part_a_step").click().run()
    assert not at.exception, f"Error clicking Step button: {at.exception}"
    assert at.session_state.part_a_iteration_slider == 1

    # Click Step again -> step 2
    at.button(key="part_a_step").click().run()
    assert not at.exception
    assert at.session_state.part_a_iteration_slider == 2

    # 2. Reset button interaction: return to step 0
    at.button(key="part_a_reset").click().run()
    assert not at.exception, f"Error clicking Reset button: {at.exception}"
    assert at.session_state.part_a_iteration_slider == 0

    # 3. Play and Pause buttons
    at.button(key="part_a_play").click().run()
    assert not at.exception, f"Error clicking Play button: {at.exception}"

    at.button(key="part_a_pause").click().run()
    assert not at.exception, f"Error clicking Pause button: {at.exception}"
    assert at.session_state.is_playing is False

    # 4. Surface selection: changing surface resets iteration slider to 0
    surface_select = at.selectbox(key="part_a_surface_select")
    assert surface_select is not None
    surface_select.select("L1: x² + 10y²").run()
    assert not at.exception, f"Error selecting surface: {at.exception}"
    assert at.session_state.part_a_iteration_slider == 0

    # 5. Multiple optimizers selection in Part A
    opt_multiselect = at.multiselect(key="part_a_optimizer_multiselect")
    assert opt_multiselect is not None
    opt_multiselect.select("NAG").select("RMSProp").select("AdamW").run()
    assert not at.exception, f"Error selecting multiple optimizers: {at.exception}"

    # Step on multi-optimizer setup
    at.button(key="part_a_step").click().run()
    assert not at.exception
    assert at.session_state.part_a_iteration_slider == 1

    # 6. Part B: Configure training and run benchmark
    epochs_slider = at.slider(key="nn_epochs_slider")
    assert epochs_slider is not None
    epochs_slider.set_value(10).run()
    assert not at.exception, f"Error setting epochs slider: {at.exception}"

    train_btn = at.button(key="nn_start_train_btn")
    assert train_btn is not None
    train_btn.click().run()
    assert not at.exception, f"Error during neural network training execution: {at.exception}"
    assert len(at.session_state.nn_histories) > 0
