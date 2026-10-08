"""Static assumptions. ALL NUMBERS ARE PLACEHOLDERS - replace with figures from
CACP cost-of-cultivation reports / Maharashtra agriculture dept / ICRISAT before the demo.
Units: yield in quintals per hectare, cost in Rs per hectare (all-in), prices in Rs per quintal.
"""

CROPS = {
    "Paddy":     dict(kw=["paddy"],                    yield_q_ha=25,  cost_ha=55000,  months=4,  seasons=["Kharif"]),
    "Maize":     dict(kw=["maize"],                    yield_q_ha=30,  cost_ha=40000,  months=4,  seasons=["Kharif", "Rabi"]),
    "Jowar":     dict(kw=["jowar"],                    yield_q_ha=12,  cost_ha=28000,  months=4,  seasons=["Kharif", "Rabi"]),
    "Bajra":     dict(kw=["bajra"],                    yield_q_ha=14,  cost_ha=25000,  months=4,  seasons=["Kharif"]),
    "Wheat":     dict(kw=["wheat"],                    yield_q_ha=25,  cost_ha=40000,  months=4,  seasons=["Rabi"]),
    "Arhar":     dict(kw=["arhar", "red gram"],        yield_q_ha=8,   cost_ha=35000,  months=6,  seasons=["Kharif"]),
    "Moong":     dict(kw=["green gram", "moong"],      yield_q_ha=5,   cost_ha=25000,  months=3,  seasons=["Kharif", "Summer"]),
    "Urad":      dict(kw=["black gram", "urad"],       yield_q_ha=5,   cost_ha=25000,  months=3,  seasons=["Kharif", "Summer"]),
    "Gram":      dict(kw=["bengal gram", "chana"],     yield_q_ha=10,  cost_ha=30000,  months=4,  seasons=["Rabi"]),
    "Groundnut": dict(kw=["groundnut"],                yield_q_ha=15,  cost_ha=50000,  months=4,  seasons=["Kharif", "Summer"]),
    "Soybean":   dict(kw=["soyabean", "soybean"],      yield_q_ha=15,  cost_ha=38000,  months=4,  seasons=["Kharif"]),
    "Safflower": dict(kw=["safflower"],                yield_q_ha=7,   cost_ha=25000,  months=5,  seasons=["Rabi"]),
    "Cotton":    dict(kw=["cotton"],                   yield_q_ha=12,  cost_ha=60000,  months=6,  seasons=["Kharif"]),
    "Onion":     dict(kw=["onion"],                    yield_q_ha=200, cost_ha=120000, months=4,  seasons=["Kharif", "Rabi"]),
    # Sugarcane is bought by mills at a government-set FRP, so it usually has no mandi series.
    # fixed_price_q is a placeholder: update it to the current FRP.
    "Sugarcane": dict(kw=["sugarcane"],                yield_q_ha=800, cost_ha=150000, months=12, seasons=["Annual"],
                      fixed_price_q=350),
}

# Approximate district centroids (lat, lon), used only to estimate transport distance.
DISTRICT_COORDS = {
    "Pune": (18.52, 73.86), "Nashik": (20.00, 73.79), "Nagpur": (21.15, 79.09),
    "Aurangabad": (19.88, 75.34), "Chhatrapati Sambhajinagar": (19.88, 75.34),
    "Kolhapur": (16.70, 74.24), "Solapur": (17.66, 75.91),
    "Ahmednagar": (19.10, 74.74), "Ahilyanagar": (19.10, 74.74),
    "Satara": (17.68, 74.00), "Sangli": (16.85, 74.57), "Jalgaon": (21.00, 75.57),
    "Amravati": (20.93, 77.75), "Akola": (20.71, 77.00), "Latur": (18.40, 76.58),
    "Nanded": (19.16, 77.31), "Yavatmal": (20.39, 78.12), "Dhule": (20.90, 74.78),
    "Beed": (18.99, 75.76), "Osmanabad": (18.18, 76.04), "Dharashiv": (18.18, 76.04),
    "Buldhana": (20.53, 76.18), "Wardha": (20.75, 78.60), "Parbhani": (19.27, 76.77),
    "Jalna": (19.84, 75.88), "Washim": (20.11, 77.13), "Hingoli": (19.72, 77.15),
    "Chandrapur": (19.95, 79.30), "Ratnagiri": (16.99, 73.30),
}

# ---- Agronomic rules of thumb (NOT a trained model). Verify with a local KVK / agriculture officer. ----
SOIL_GOOD = {
    "Paddy": ["Alluvial", "Laterite", "Black"], "Maize": ["Alluvial", "Red", "Medium black"],
    "Jowar": ["Black", "Medium black"], "Bajra": ["Sandy", "Red", "Medium black"],
    "Wheat": ["Black", "Medium black", "Alluvial"], "Arhar": ["Black", "Medium black", "Red"],
    "Moong": ["Alluvial", "Red", "Medium black"], "Urad": ["Black", "Medium black", "Alluvial"],
    "Gram": ["Black", "Medium black"], "Groundnut": ["Red", "Sandy", "Medium black"],
    "Soybean": ["Black", "Medium black"], "Safflower": ["Black", "Medium black"],
    "Cotton": ["Black", "Medium black"], "Onion": ["Alluvial", "Red", "Medium black"],
    "Sugarcane": ["Black", "Alluvial", "Medium black"],
}
SOIL_POOR = {
    "Paddy": ["Sandy"], "Maize": ["Sandy"], "Jowar": ["Laterite"], "Bajra": ["Black"],
    "Wheat": ["Sandy", "Laterite"], "Arhar": ["Sandy"], "Moong": [], "Urad": ["Sandy"],
    "Gram": ["Sandy", "Laterite"], "Groundnut": ["Black"], "Soybean": ["Sandy", "Laterite"],
    "Safflower": ["Sandy", "Laterite"], "Cotton": ["Sandy", "Laterite"], "Onion": ["Black"],
    "Sugarcane": ["Sandy", "Laterite"],
}
# 1 = low, 2 = medium, 3 = high water need
WATER_NEED = {
    "Paddy": 3, "Sugarcane": 3, "Maize": 2, "Wheat": 2, "Onion": 2, "Groundnut": 2, "Cotton": 2,
    "Jowar": 1, "Bajra": 1, "Arhar": 1, "Moong": 1, "Urad": 1, "Gram": 1, "Safflower": 1, "Soybean": 1,
}
FAMILY = {
    "Paddy": "cereal", "Maize": "cereal", "Jowar": "cereal", "Bajra": "cereal", "Wheat": "cereal",
    "Arhar": "pulse", "Moong": "pulse", "Urad": "pulse", "Gram": "pulse",
    "Groundnut": "oilseed", "Soybean": "oilseed", "Safflower": "oilseed",
    "Cotton": "cash", "Onion": "cash", "Sugarcane": "cash",
}
LEGUMES = {"Arhar", "Moong", "Urad", "Gram", "Groundnut", "Soybean"}
