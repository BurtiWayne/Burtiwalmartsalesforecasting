.PHONY: install dashboard pipeline test lint notebook clean

install:
	pip install -r requirements-dev.txt

dashboard:
	streamlit run app/streamlit_app.py

pipeline:
	python run_pipeline.py

test:
	pytest tests/ -v

lint:
	ruff check src app tests run_pipeline.py

notebook:
	jupyter lab notebooks/Walmart_Sales_Analysis_and_Forecasting.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
