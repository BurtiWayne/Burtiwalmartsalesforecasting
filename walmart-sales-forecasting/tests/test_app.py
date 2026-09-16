"""Smoke test: every dashboard page must render without raising."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

PAGES = [
    "Overview",
    "Seasonality",
    "Economic Factors",
    "Store Performance",
    "Forecast",
    "Model Details",
]


def render(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    at.radio[0].set_value(page).run()
    return at


def test_all_pages_render():
    for page in PAGES:
        at = render(page)
        assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"


if __name__ == "__main__":
    for p in PAGES:
        at = render(p)
        print(f"{'FAIL' if at.exception else 'OK':5} {p}")
        for e in at.exception:
            print("   ", e.value)
