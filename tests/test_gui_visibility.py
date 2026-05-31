"""Visibility checks for the simplified desktop GUI."""

from __future__ import annotations

import os
import sys


def test_desktop_gui_main_menu_is_visible_and_simplified(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from PyQt5.QtWidgets import QApplication
    from desktop_gui import FancyGUI

    app = QApplication.instance() or QApplication([])
    gui = FancyGUI()
    try:
        gui.show()
        app.processEvents()

        button_texts = [button.text() for button in gui._menu_buttons]
        joined = "\n".join(button_texts)
        assert "Track Cells" in joined
        assert "Train Cell Line" in joined
        assert "Train Phenotype" not in joined
        assert "Public data search or local files" in joined
        assert "Analyze Results" in joined
        assert "Generate Test" not in joined
        assert "Synthetic" not in joined

        pixmap = gui.grab()
        assert pixmap.width() >= 900
        assert pixmap.height() >= 600

        image = pixmap.toImage()
        sample_points = [
            (image.width() // 2, image.height() // 2),
            (image.width() // 4, image.height() // 3),
            (3 * image.width() // 4, image.height() // 3),
        ]
        colors = {
            image.pixelColor(x, y).getRgb()[:3]
            for x, y in sample_points
        }
        assert len(colors) > 1

        artifact_dir = os.path.join(repo_root, "local_test_outputs", "gui_visibility")
        os.makedirs(artifact_dir, exist_ok=True)
        assert pixmap.save(os.path.join(artifact_dir, "main_menu.png"))
    finally:
        gui.cleanup()
