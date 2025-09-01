# trainpeek_pro_app.py
# Neue Codebasis: Streamlit-App mit Wochen- & Monatskalender, Phasen, Wettkämpfen, PMC & Wochenlast
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, timedelta

st.set_page_config(page_title="TrainPeek Pro", page_icon="🏃‍♂️", layout="wide")

DATA_DIR = Path("data")
WORKOUTS_FILE = DATA_DIR / "workouts.csv"
PLAN_FILE = DATA_DIR / "plan.csv"

WORKOUT_COLS = ["date","sport","title","duration_min","distance_km","rpe","tss","notes"]
PLAN_COLS    = ["kind","title","start_date","end_date","sport","priority","color","notes"]

SPORTS = ["Run","Bike","Swim","Strength","Other"]
PRIOS  = ["A","B","C"]

def ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    import numpy as np
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)

    # fehlende Spalten ergänzen & Reihenfolge sichern
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    df = df[columns].copy()

    # leere Strings -> NaN (sonst stolpert to_datetime)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Zahlenfelder
    for c in ["duration_min","distance_km","rpe","tss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Textfelder
    for c in ["title","sport","priority","notes","kind","color"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    # Datumsfelder ROBUST parsen (erlaubt ISO 2025-09-25 und 25.09.2025)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.date

    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", dayfirst=True).dt.date
    if "end_date" in df.columns:
        df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce", dayfirst=True).dt.date

    # Unbrauchbare Plan-Zeilen entfernen (ohne Start/Ende)
    if {"start_date","end_date"}.issubset(df.columns):
        df = df[~df["start_date"].isna() & ~df["end_date"].isna()].copy()

    return df


def save_csv(df: pd.DataFrame, path: Path):
    ensure_dir()
    df.to_csv(path, index=False)

def training_load(row) -> float:
    tss = row.get("tss", np.nan)
    if pd.notna(tss) and float(tss) > 0:
        return float(tss)
    return (row.get("duration_min") or 0) * (row.get("rpe") or 0)

def daily_load(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","TL"])
    tmp = df.copy()
    tmp["TL"] = tmp.apply(training_load, axis=1)
    return tmp.groupby("date", as_index=False)["TL"].sum().sort_values("date")

def exp_ema(values: np.ndarray, tau_days: float) -> np.ndarray:
    alpha = 1 - np.exp(-1 / tau_days)
    ema, prev = [], 0.0
    for x in values:
        prev = prev + (x - prev) * alpha
        ema.append(prev)
    return np.array(ema)

def compute_pmc(daily: pd.DataFrame, tau_atl=7.0, tau_ctl=42.0) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date","TL","ATL","CTL","TSB"])
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    frame = pd.DataFrame({"date": dates})
    merged = frame.merge(daily, on="date", how="left").fillna({"TL": 0.0})
    ATL = exp_ema(merged["TL"].to_numpy(float), tau_atl)
    CTL = exp_ema(merged["TL"].to_numpy(float), tau_ctl)
    merged["ATL"] = ATL
    merged["CTL"] = CTL
    merged["TSB"] = CTL - ATL
    return merged

def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())

def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["week_start","sessions","duration_min","distance_km","TL"])
    tmp = df.copy()
    tmp["week_start"] = tmp["date"].apply(week_start)
    tmp["TL"] = tmp.apply(training_load, axis=1)
    return tmp.groupby("week_start", as_index=False).agg(
        sessions=("date","count"),
        duration_min=("duration_min","sum"),
        distance_km=("distance_km","sum"),
        TL=("TL","sum"),
    ).sort_values("week_start")

def week_days(ref: date) -> list[date]:
    ws = week_start(ref)
    return [ws + timedelta(days=i) for i in range(7)]

# ---------------------- UI ----------------------
def main():
    ensure_dir()
    st.title("🏃‍♂️ TrainPeek Pro – mobil & Kalender")

    wdf = load_csv(WORKOUTS_FILE, WORKOUT_COLS)
    pdf = load_csv(PLAN_FILE, PLAN_COLS)

    with st.sidebar:
        st.header("Workout hinzufügen")
        d_val = st.date_input("Datum", value=date.today())
        sport = st.selectbox("Sport", SPORTS, index=0)
        title = st.text_input("Titel", value="Workout")
        dur = st.number_input("Dauer (Min)", 0, step=5, value=60)
        dist = st.number_input("Distanz (km)", 0.0, step=0.5, value=0.0)
        rpe = st.slider("RPE (1–10)", 1, 10, 6)
        tss = st.number_input("TSS (optional)", 0.0, step=5.0, value=0.0)
        notes = st.text_area("Notizen", height=60)
        if st.button("Speichern"):
            new = pd.DataFrame([{
                "date": d_val, "sport": sport, "title": title,
                "duration_min": dur, "distance_km": dist if dist>0 else np.nan,
                "rpe": rpe, "tss": tss if tss>0 else np.nan, "notes": notes
            }])
            wdf = pd.concat([wdf,new], ignore_index=True)
            save_csv(wdf, WORKOUTS_FILE)
            st.success("Workout gespeichert!")

    tab1, tab2 = st.tabs(["📅 Übersicht","📈 Dashboard"])

    with tab1:
        ref = date.today()
        days = week_days(ref)
        cols = st.columns(7)
        for i,d in enumerate(days):
            with cols[i]:
                st.markdown(f"**{d.strftime('%a %d.%m.')}**")
                workouts = wdf[wdf["date"]==d]
                for _, w in workouts.iterrows():
                    st.markdown(f"- {w['title']} ({w['sport']}, {w['duration_min']} min)")

    with tab2:
        wk = weekly_summary(wdf)
        if wk.empty:
            st.info("Noch keine Workouts.")
        else:
            st.dataframe(wk, use_container_width=True)
            fig, ax = plt.subplots()
            ax.bar(wk["week_start"].astype(str), wk["TL"])
            st.pyplot(fig)

        daily = daily_load(wdf)
        pmc = compute_pmc(daily)
        if not pmc.empty:
            fig2, ax2 = plt.subplots()
            ax2.plot(pmc["date"], pmc["CTL"], label="CTL")
            ax2.plot(pmc["date"], pmc["ATL"], label="ATL")
            ax2.legend()
            st.pyplot(fig2)

if __name__ == "__main__":
    main()
