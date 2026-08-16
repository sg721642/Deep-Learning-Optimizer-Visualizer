"""
Integration test for Streamlit app to verify no DuplicateElementId or runtime crashes occur during interactions.
"""
import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_part_a_and_b():
    """Run simulated Streamlit test across Part A and Part B."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit app raised an exception on load: {at.exception}"

    # Verify title
    assert len(at.tabs) >= 5

    # Interact with Part A: Change surface and learning rate
    surface_select = at.selectbox(key="part_a_surface_select")
    assert surface_select is not None
    surface_select.select("L1: x² + 10y²").run()
    assert not at.exception, f"Error selecting surface: {at.exception}"

    # Click Step button
    btn_step = at.button(key="btn_step")
    assert btn_step is not None
    btn_step.click().run()
    assert not at.exception, f"Error clicking step button: {at.exception}"

    # Click Reset button
    btn_reset = at.button(key="btn_reset")
    assert btn_reset is not None
    btn_reset.click().run()
    assert not at.exception, f"Error clicking reset button: {at.exception}"

    # Interact with Part B: Configure training
    epochs_slider = at.slider(key="nn_epochs_slider")
    assert epochs_slider is not None
    # Set to small epoch for fast integration test
    epochs_slider.set_value(10).run()
    assert not at.exception, f"Error setting epochs slider: {at.exception}"

    # Click start training button
    train_btn = at.button(key="nn_start_train_btn")
    assert train_btn is not None
    train_btn.click().run()
    assert not at.exception, f"Error during neural network training execution: {at.exception}"
