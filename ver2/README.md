# Crop advisor for Maharashtra farmers

Run:
    pip install -r requirements.txt
    streamlit run app.py

The farmer answers questions (land, soil, water, past crops, budget, loan/subsidy, target income,
storage, cash need, transport). The app checks agronomic suitability (soil, water, rotation rules of thumb)
and economic feasibility (price forecast, cost, transport, storage) and shows the best 3 crops.

Real prices: put Agmarknet CSV(s) in data/ named prices*.csv (state, district, market, commodity, date, modal price).
Without a file the app runs on SYNTHETIC demo prices and shows a warning.

crop_data.py: yield, cost, sugarcane price and the soil/water rules are PLACEHOLDERS or rules of thumb.
Replace/verify before demoing.
