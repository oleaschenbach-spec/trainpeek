# trainpeek_pro_app.py
# Kalender (Woche/Monat), Phasen (Grundlage/Aufbau/Spitze/Taper/Erholung),
# Wochen-Planer und automatische TSS-Berechnung aus Dauer + HR oder (nur Bike) Watt.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, timedelta
import json

st.set_page_config(page_title="TrainPeek Pro", page_icon="🏃‍♂️", layout="wide")

DATA_DIR = Path("data")
WORKOUTS_FILE = DATA_DIR / "workouts.csv"
PLAN_FILE = DATA_DIR / "plan.csv"
SETTINGS_FILE = DATA_DIR / "settings.json"

# NEU: avg_hr, avg_power
WORKOUT_COLS = [
    "date","sport","title","duration_min","distance_km","rpe","tss","avg_hr","avg_power","notes"
]
PLAN_COLS = [
    "kind","phase_type","title","start_date","end_date","sport","priority","color","notes"
]  # kind: phase|race

SPORTS = ["Run","Bike","Swim","Strength","Other"]
PRIOS  = ["A","B","C"]

PHASE_TYPES = {
    "Grundlage": {"color": "#E3F2FD"},
    "Aufbau":    {"color": "#E8F5E9"},
    "Spitze":    {"color": "#FFF3E0"},
    "Taper":     {"color": "#F3E5F5"},
    "Erholung":  {"color": "#FFEBEE"},
}

# -------------------- Storage helpers --------------------
def ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_settings():
    ensure_dir()
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"ftp_watt": None, "lthr_bpm": None}
    return {"ftp_watt": None, "lthr_bpm": None}

def save_settings(s):
    ensure_dir()
    SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)
    # fehlende Spalten ergänzen
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    df = df[columns].copy()

    # leere Strings -> NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # numerisch parsen
    for c in ["duration_min","distance_km","rpe","tss","avg_hr","avg_power"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # textfelder
    for c in ["title","sport","priority","notes","kind","color","phase_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    # datum robust (DD.MM.YYYY erlaubt)
    for c in ["date","start_date","end_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.date

    # plan-zeilen ohne datumsrange raus
    if {"start_date","end_date"}.issubset(df.columns):
        df = df[df["start_date"].notna() & df["end_date"].notna()].copy()

    return df

def save_csv(df: pd.DataFrame, path: Path):
    ensure_dir()
    df.to_csv(path, index=False)

# -------------------- TSS logic --------------------
def auto_tss_from_metrics(row: pd.Series, settings: dict) -> float | None:
    """
    Versucht TSS zu berechnen:
    - Bike + avg_power + ftp_watt -> powerTSS
    - avg_hr + lthr_bpm -> hrTSS
    - sonst None
    Formel:
        TSS = Dauer_h * (Intensität)^2 * 100
        Intensität_power = avg_power / FTP
        Intensität_hr    = avg_hr / LTHR
    """
    dur_min = float(row.get("duration_min") or 0.0)
    if dur_min <= 0:
        return None
    dur_h = dur_min / 60.0

    # Power-basiert (nur Bike)
    if str(r

