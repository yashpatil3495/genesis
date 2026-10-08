import glob
import logging
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from crop_data import (CROPS, DISTRICT_COORDS, SOIL_GOOD, SOIL_POOR, WATER_NEED, FAMILY, LEGUMES)

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# ---- assumptions (placeholders: replace with sourced figures) ----
TRANSPORT_RATES = {"Own vehicle": 0.15, "Tractor / trolley": 0.20,
                   "Hired tempo / truck": 0.30, "Shared / FPO transport": 0.18}  # Rs per quintal per km
STORAGE_RS_Q_MONTH = 10     # Rs per quintal per month
LOAN_RATE = 0.07            # yearly interest on crop loan
SELL_WINDOW_WEEKS = 12
ACRE_TO_HA = 0.4047
SOILS = {"Black (deep, heavy)": "Black", "Medium black": "Medium black", "Red": "Red",
         "Laterite": "Laterite", "Alluvial / loamy": "Alluvial", "Sandy / light": "Sandy", "Not sure": "Not sure"}
IRRIG_CAP = {"Rainfed (no irrigation)": 1.0, "Borewell / well": 2.0, "Drip": 2.5, "Canal": 3.0}
NONE = "None / fallow"

st.set_page_config(page_title="Crop advisor for Maharashtra farmers", layout="wide")


# ================================================================ data
def km(a, b):
    if a == b:
        return 0.0
    if a not in DISTRICT_COORDS or b not in DISTRICT_COORDS:
        return 100.0
    (la1, lo1), (la2, lo2) = DISTRICT_COORDS[a], DISTRICT_COORDS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dl = math.radians(lo2 - lo1)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h)) * 1.3


def make_demo_data():
    """SYNTHETIC prices so the app runs before real data is ready. Never present as real."""
    rng = np.random.default_rng(42)
    base = {"Paddy": 2300, "Maize": 2200, "Jowar": 3200, "Bajra": 2400, "Wheat": 2500,
            "Arhar": 7500, "Moong": 8000, "Urad": 7000, "Gram": 5500, "Groundnut": 6000,
            "Soybean": 4600, "Safflower": 5800, "Cotton": 7000, "Onion": 1800}
    phase = {c: rng.integers(0, 365) for c in base}
    markets = [("Pune", "Pune"), ("Nashik", "Lasalgaon"), ("Nagpur", "Nagpur"),
               ("Kolhapur", "Kolhapur"), ("Latur", "Latur"), ("Solapur", "Solapur")]
    offsets = {m: rng.uniform(-0.05, 0.05) for _, m in markets}
    dates = pd.date_range("2023-01-01", "2026-09-30", freq="W")
    rows = []
    for crop, b in base.items():
        doy = dates.dayofyear.values
        for district, market in markets:
            season = 0.12 * np.sin(2 * np.pi * (doy - phase[crop]) / 365)
            trend = np.linspace(0, 0.08, len(dates))
            noise = rng.normal(0, 0.03, len(dates))
            price = b * (1 + season + trend + noise + offsets[market])
            rows.append(pd.DataFrame({"date": dates, "district": district, "market": market,
                                      "crop": crop, "modal": price}))
    return pd.concat(rows, ignore_index=True)


def pick(df, key):
    for c in df.columns:
        if key in c:
            return c
    raise KeyError(f"No column containing '{key}' in {list(df.columns)}")


@st.cache_data(show_spinner="Loading prices...")
def load_prices():
    files = sorted(glob.glob("data/prices*.csv"))
    if not files:
        return make_demo_data(), True
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "state" in df.columns:
        df = df[df["state"].astype(str).str.strip().str.lower() == "maharashtra"]
    out = pd.DataFrame({
        "date": pd.to_datetime(df[pick(df, "date")], dayfirst=True, errors="coerce"),
        "district": df[pick(df, "district")].astype(str).str.strip().str.title(),
        "market": df[pick(df, "market")].astype(str).str.strip(),
        "commodity": df[pick(df, "commodity")].astype(str).str.lower(),
        "modal": pd.to_numeric(df[pick(df, "modal")], errors="coerce"),
    }).dropna(subset=["date", "modal"])
    out = out[out["modal"] > 0]
    out["crop"] = None
    for crop, spec in CROPS.items():
        pat = "|".join(re.escape(k) for k in spec["kw"])
        mask = out["commodity"].str.contains(pat, regex=True) & out["crop"].isna()
        out.loc[mask, "crop"] = crop
    out = out.dropna(subset=["crop"]).drop(columns="commodity")
    return out.reset_index(drop=True), False


def forecast_series(ws, weeks):
    last = ws["ds"].max()
    future = pd.date_range(last + pd.Timedelta(weeks=1), periods=weeks, freq="W")
    if len(ws) >= 60:
        try:
            from prophet import Prophet
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, interval_width=0.8)
            m.fit(ws)
            fc = m.predict(pd.DataFrame({"ds": future}))
            return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                columns={"yhat_lower": "low", "yhat_upper": "high"}).reset_index(drop=True)
        except Exception:
            pass
    vals = []
    for d in future:
        ref = ws[(ws.ds >= d - pd.Timedelta(weeks=54)) & (ws.ds <= d - pd.Timedelta(weeks=50))]["y"]
        vals.append(ref.mean() if len(ref) else ws["y"].iloc[-1])
    vals = np.array(vals)
    return pd.DataFrame({"ds": future, "yhat": vals, "low": vals * 0.85, "high": vals * 1.15})


@st.cache_data(show_spinner="Preparing price forecasts (first load takes a minute)...")
def run_forecasts(prices):
    out = {}
    for crop, spec in CROPS.items():
        weeks = math.ceil(spec["months"] * 30.4 / 7) + SELL_WINDOW_WEEKS
        d = prices[prices["crop"] == crop]
        ws = (d.set_index("date")["modal"].resample("W").mean().interpolate(limit=4).dropna()
              .reset_index().rename(columns={"date": "ds", "modal": "y"}))
        if len(ws) >= 8:
            out[crop] = dict(fc=forecast_series(ws, weeks), hist=ws.tail(52),
                             now=float(ws["y"].tail(4).mean()), fixed=False)
        elif "fixed_price_q" in spec:
            p = spec["fixed_price_q"]
            future = pd.date_range(pd.Timestamp.today().normalize(), periods=weeks, freq="W")
            out[crop] = dict(fc=pd.DataFrame({"ds": future, "yhat": p, "low": p * 0.95, "high": p * 1.05}),
                             hist=None, now=float(p), fixed=True)
        else:
            out[crop] = None
    return out


# ================================================================ agronomy rules
def soil_factor(crop, soil):
    if soil == "Not sure":
        return 0.85
    if soil in SOIL_GOOD[crop]:
        return 1.0
    if soil in SOIL_POOR[crop]:
        return 0.6
    return 0.85


def water_factor(crop, irrigation, season):
    cap = IRRIG_CAP[irrigation] + (1.0 if season == "Kharif" else 0.0)  # monsoon adds water
    gap = WATER_NEED[crop] - cap
    if gap <= 0:
        return 1.0
    if gap <= 0.5:
        return 0.85
    if gap <= 1.0:
        return 0.6
    return 0.3


def rotation(crop, past):
    """past = crops grown, most recent first. Returns (yield factor, note)."""
    if not past:
        return 1.0, ""
    count = sum(1 for p in past if p == crop)
    if count >= 2:
        return 0.75, f"{crop} was grown here repeatedly; pests and soil tiredness build up"
    if past[0] == crop:
        return 0.85, f"{crop} was grown last season, so rotation is poor"
    if count == 1:
        return 0.92, f"{crop} was grown here recently"
    if crop in LEGUMES and past[0] not in LEGUMES:
        return 1.05, f"{crop} fixes nitrogen and follows {past[0]} well"
    if FAMILY[crop] == FAMILY[past[0]] and crop not in LEGUMES:
        return 0.95, f"Same crop family as last season's {past[0]}"
    return 1.0, f"Good rotation after {past[0]}"


# ================================================================ economics
def market_choice(d, district, max_km, rate):
    latest = d["date"].max()
    r = d[d["date"] >= latest - pd.Timedelta(days=45)]
    if r.empty:
        r = d
    g = r.groupby(["district", "market"])["modal"].mean().reset_index()
    g["dist_km"] = g["district"].apply(lambda x: max(10.0, km(district, x)))
    g["transport"] = g["dist_km"] * rate
    g["net_now"] = g["modal"] - g["transport"]
    within = g[g["dist_km"] <= max_km]
    far = within.empty
    best = g.sort_values("dist_km").iloc[0] if far else within.sort_values("net_now", ascending=False).iloc[0]
    return best, best["modal"] / r["modal"].mean(), far


def analyse(inp, forecasts, prices, assume):
    season = "Summer" if inp["season"] == "Zaid" else inp["season"]
    past = [p for p in inp["past"] if p != NONE]
    rate = inp["rate_override"] / 100 if inp["rate_override"] > 0 else TRANSPORT_RATES[inp["transport"]]
    wait_ok = inp["storage"] == "Yes" and inp["cash"].startswith("I can wait")
    use_loan = inp["finance"] in ("Crop loan", "Both")
    use_sub = inp["finance"] in ("Subsidy", "Both")
    rows, excluded = [], []
    for crop, spec in CROPS.items():
        if season not in spec["seasons"] and "Annual" not in spec["seasons"]:
            continue
        f = forecasts.get(crop)
        if f is None:
            continue
        sf = soil_factor(crop, inp["soil"])
        wf = water_factor(crop, inp["irrigation"], inp["season"])
        rf, rnote = rotation(crop, past)
        suit = sf * wf * rf
        if min(1.0, suit) < 0.4:
            why = "soil" if sf < wf else "water"
            excluded.append(f"{crop} ({why} not suitable)")
            continue
        fc = f["fc"]
        h_idx = min(math.ceil(spec["months"] * 30.4 / 7) - 1, len(fc) - 1)
        d = prices[prices["crop"] == crop]
        far = False
        if f["fixed"] or d.empty:
            ratio, transport, mkt, dist = 1.0, 0.0, "Sugar mill / assured price", 0.0
        else:
            b, ratio, far = market_choice(d, inp["district"], inp["max_km"], rate)
            transport, dist = float(b["transport"]), float(b["dist_km"])
            mkt = f'{b["market"]} ({b["district"]})'
        y_base, cost_ha = float(assume.loc[crop, "yield_q_ha"]), float(assume.loc[crop, "cost_ha"])
        planted = inp["land_ha"] if inp["budget"] <= 0 else min(inp["land_ha"], inp["budget"] / cost_ha)
        qty = y_base * suit * planted
        cost_total = cost_ha * planted
        if use_loan:
            cost_total += cost_total * LOAN_RATE * spec["months"] / 12
        if use_sub:
            cost_total -= min(inp["subsidy"], cost_ha * planted)

        def net_price(w, col):
            storage = STORAGE_RS_Q_MONTH * max(0, w - h_idx) / 4.33
            return fc.iloc[w][col] * ratio - transport - storage

        if wait_ok:
            last_w = min(h_idx + SELL_WINDOW_WEEKS, len(fc) - 1)
            w = max(range(h_idx, last_w + 1), key=lambda i: net_price(i, "yhat"))
        else:
            w = h_idx
        p_low, p_mid, p_high = [qty * net_price(w, c) - cost_total for c in ("low", "yhat", "high")]
        gain = (net_price(w, "yhat") / net_price(h_idx, "yhat") - 1) * 100
        sell_date = fc.iloc[w]["ds"].date()
        if w == h_idx:
            if wait_ok:
                why = ""
            elif inp["storage"] == "No":
                why = ", since you have no storage"
            else:
                why = ", since you need cash at harvest"
            sell_txt = f"Sell soon after harvest (around {sell_date}){why}."
        else:
            sell_txt = f"Store and sell around {sell_date} (about {gain:+.1f}% vs selling at harvest, after storage cost)."

        notes = []
        if inp["soil"] != "Not sure":
            notes.append(f"{inp['soil']} soil " + ("suits this crop" if sf == 1.0 else
                         "is only fair for this crop" if sf < 1.0 and sf > 0.6 else "is a poor match, so yield is cut"))
        notes.append("Your water source is enough" if wf == 1.0 else "Water may be short, so yield is reduced")
        if rnote:
            notes.append(rnote)
        notes.append(f"Expected profit Rs {p_mid:,.0f} on {planted:.1f} ha; Rs {p_low:,.0f} if prices come in low")
        notes.append(f"Best market: {mkt}" + (f", {dist:.0f} km away" if dist else "") +
                     (" (nearest one, beyond your travel limit)" if far else ""))
        if planted < inp["land_ha"] - 1e-9:
            notes.append(f"Your budget covers only {planted / inp['land_ha'] * 100:.0f}% of your land for this crop")
        meets = p_mid >= inp["target"] if inp["target"] > 0 else True
        if inp["target"] > 0 and not meets:
            notes.append("Does not reach your target income")
        rows.append({
            "Crop": crop, "Suitability %": round(min(1.0, suit) * 100), "Planted (ha)": round(planted, 2),
            "Cost (Rs)": round(cost_total), "Profit low (Rs)": round(p_low), "Profit expected (Rs)": round(p_mid),
            "Profit high (Rs)": round(p_high), "Meets target": meets, "Best market": mkt,
            "Distance (km)": round(dist), "Sell on": str(sell_date), "Sell advice": sell_txt, "Notes": notes,
            "Rank score": 0.6 * p_mid + 0.4 * p_low, "_f": f, "_h": h_idx, "_w": w,
        })
    if not rows:
        return pd.DataFrame(), excluded
    res = pd.DataFrame(rows).sort_values(["Meets target", "Rank score"], ascending=[False, False])
    return res.reset_index(drop=True), excluded


# ================================================================ UI
prices, is_demo = load_prices()
forecasts = run_forecasts(prices)

st.title("Crop advisor: what should I grow this season?")
st.caption("Answer a few questions. We check what suits your land and what is likely to pay, then suggest your best 3 crops.")
if is_demo:
    st.warning("Showing SYNTHETIC demo prices (no file found in data/prices*.csv). Not real recommendations yet.")

base = pd.DataFrame({c: {"yield_q_ha": s["yield_q_ha"], "cost_ha": s["cost_ha"]} for c, s in CROPS.items()}).T
with st.expander("Advanced: edit crop yield (quintal/ha) and cost (Rs/ha) assumptions"):
    assume = st.data_editor(base, key="assume")

districts = sorted({k for k in DISTRICT_COORDS if k not in ("Aurangabad", "Osmanabad", "Ahmednagar")})
with st.form("farm"):
    st.subheader("1. Your land")
    c1, c2, c3, c4 = st.columns(4)
    district = c1.selectbox("District", districts, index=districts.index("Pune"))
    c1.text_input("Village / Tehsil (optional)", help="Not used yet. District decides markets and prices.")
    area = c2.number_input("Land area", 0.1, 500.0, 2.0, 0.5)
    unit = c3.selectbox("Unit", ["acres", "hectares"])
    season = c4.selectbox("Season", ["Kharif", "Rabi", "Zaid"],
                          help="Kharif: monsoon. Rabi: winter. Zaid: summer.")

    st.subheader("2. Soil, water and past crops")
    c1, c2 = st.columns(2)
    soil_label = c1.selectbox("Soil type", list(SOILS))
    irrigation = c2.selectbox("Main water source", list(IRRIG_CAP))
    c1, c2, c3 = st.columns(3)
    past1 = c1.selectbox("Last season's crop", [NONE] + list(CROPS))
    past2 = c2.selectbox("Season before that", [NONE] + list(CROPS))
    past3 = c3.selectbox("And before that", [NONE] + list(CROPS))

    st.subheader("3. Money")
    c1, c2, c3, c4 = st.columns(4)
    budget = c1.number_input("Max budget for this season (Rs, 0 = no limit)", 0, 10_000_000, 0, 5000)
    finance = c2.selectbox("Loan or subsidy?", ["Neither", "Crop loan", "Subsidy", "Both"])
    subsidy = c3.number_input("Expected subsidy amount (Rs, if any)", 0, 10_000_000, 0, 1000)
    target = c4.number_input("Minimum income you want (Rs, 0 = none)", 0, 100_000_000, 0, 10000)

    st.subheader("4. Selling")
    c1, c2, c3 = st.columns(3)
    storage = c1.radio("Do you have storage?", ["Yes", "No"], horizontal=True)
    cash = c2.radio("At harvest", ["I need cash right away", "I can wait for a better price"])
    max_km = c3.slider("How far can you take the crop? (km)", 10, 500, 100, 10)
    c1, c2 = st.columns(2)
    transport = c1.selectbox("Transport you can use", list(TRANSPORT_RATES))
    rate_override = c2.number_input("Your cost to carry 1 quintal 100 km (Rs, 0 = use estimate)", 0, 2000, 0, 5)
    submitted = st.form_submit_button("Get my recommendation", type="primary")

if submitted:
    st.session_state["inp"] = dict(
        district=district, land_ha=area * (ACRE_TO_HA if unit == "acres" else 1.0), season=season,
        soil=SOILS[soil_label], irrigation=irrigation, past=[past1, past2, past3], budget=float(budget),
        finance=finance, subsidy=float(subsidy), target=float(target), storage=storage, cash=cash,
        max_km=float(max_km), transport=transport, rate_override=float(rate_override))

if "inp" in st.session_state:
    inp = st.session_state["inp"]
    res, excluded = analyse(inp, forecasts, prices, assume)
    st.divider()
    if res.empty:
        st.error("No crop fits these answers. Try a different season or water source.")
        st.stop()
    st.header(f"Your best crops for {inp['district']} ({inp['season']})")
    if not res["Meets target"].any() and inp["target"] > 0:
        st.warning("None of the crops is expected to reach your minimum income. Showing the best available.")
    top = res.head(3)
    cols = st.columns(len(top))
    for i, (col, (_, r)) in enumerate(zip(cols, top.iterrows())):
        with col.container(border=True):
            st.subheader(f"{i + 1}. {r['Crop']}")
            st.metric("Expected profit", f"Rs {r['Profit expected (Rs)']:,}",
                      f"Low case Rs {r['Profit low (Rs)']:,}", delta_color="off")
            st.progress(int(r["Suitability %"]), text=f"Suitability for your land: {r['Suitability %']}%")
            st.markdown("\n".join(f"- {n}" for n in r["Notes"]))
            st.info(r["Sell advice"])
    if excluded:
        st.caption("Not suggested for your plot: " + ", ".join(excluded))

    with st.expander("See full analysis (all crops, charts)"):
        show = res.drop(columns=["Notes", "Rank score", "_f", "_h", "_w"])
        st.dataframe(show, hide_index=True)
        st.bar_chart(res.set_index("Crop")[["Profit low (Rs)", "Profit expected (Rs)", "Profit high (Rs)"]])
        crop = st.selectbox("Price forecast for", res["Crop"].tolist())
        r = res[res["Crop"] == crop].iloc[0]
        f = r["_f"]
        chart = pd.DataFrame(index=pd.concat([
            f["hist"]["ds"] if f["hist"] is not None else pd.Series(dtype="datetime64[ns]"), f["fc"]["ds"]]))
        if f["hist"] is not None:
            chart.loc[f["hist"]["ds"], "History"] = f["hist"]["y"].values
        for name, c in (("Low", "low"), ("Expected", "yhat"), ("High", "high")):
            chart.loc[f["fc"]["ds"], name] = f["fc"][c].values
        st.line_chart(chart)

    st.caption("Guidance only. Suitability uses general soil, water and rotation rules of thumb, and profit uses "
               "price forecasts and editable cost assumptions. Check with your local agriculture office before deciding.")
