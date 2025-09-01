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

# Spalten inkl. Avg HR / Avg Power für Auto-TSS
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

# ---------- Timestamp-Helfer (fix für Date/Datetime-Vergleiche) ----------
def _ts(x):
    """Serie oder Skalar robust zu normalisierten Timestamps konvertieren."""
    if isinstance(x, pd.Series):
        return pd.to_datetime(x, errors="coerce").dt.normalize()
    return pd.Timestamp(x).normalize()

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

    # datum robust (DD.MM.YYYY erlaubt) -> als date speichern
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
    if str(row.get("sport","")).lower() == "bike":
        avg_power = row.get("avg_power")
        ftp = settings.get("ftp_watt")
        if pd.notna(avg_power) and avg_power and ftp and ftp > 0:
            IF = float(avg_power) / float(ftp)
            return dur_h * (IF ** 2) * 100.0

    # HR-basiert (alle Sportarten möglich)
    avg_hr = row.get("avg_hr")
    lthr = settings.get("lthr_bpm")
    if pd.notna(avg_hr) and avg_hr and lthr and lthr > 0:
        HRr = float(avg_hr) / float(lthr)
        return dur_h * (HRr ** 2) * 100.0

    return None

def training_load(row: pd.Series, settings: dict) -> float:
    # Wenn TSS schon da ist, nutze es
    tss = row.get("tss", np.nan)
    if pd.notna(tss) and float(tss) > 0:
        return float(tss)
    # sonst versuchen aus Metriken zu berechnen
    tss_auto = auto_tss_from_metrics(row, settings)
    if tss_auto is not None:
        return float(tss_auto)
    # Fallback: Dauer × RPE
    return float(row.get("duration_min") or 0) * float(row.get("rpe") or 0)

def daily_load(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["date","TL"])
    tmp = df.copy()
    tmp["TL"] = tmp.apply(lambda r: training_load(r, settings), axis=1)
    return tmp.groupby("date", as_index=False)["TL"].sum().sort_values("date")

def exp_ema(values: np.ndarray, tau_days: float) -> np.ndarray:
    alpha = 1 - np.exp(-1 / tau_days)
    ema, prev = [], 0.0
    for x in values:
        prev = prev + (x - prev) * alpha
        ema.append(prev)
    return np.array(ema, dtype=float)

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

def week_days(ref: date):
    ws = week_start(ref)
    return [ws + timedelta(days=i) for i in range(7)]

def weekly_summary(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["week_start","sessions","duration_min","distance_km","TL"])
    tmp = df.copy()
    tmp["week_start"] = tmp["date"].apply(week_start)
    tmp["TL"] = tmp.apply(lambda r: training_load(r, settings), axis=1)
    return tmp.groupby("week_start", as_index=False).agg(
        sessions=("date","count"),
        duration_min=("duration_min","sum"),
        distance_km=("distance_km","sum"),
        TL=("TL","sum"),
    ).sort_values("week_start")

# -------------------- UI bits --------------------
def phase_color(phase_type: str) -> str:
    return PHASE_TYPES.get(phase_type, {}).get("color", "#FFFDE7")

def weekly_planner_form(ref_week: date):
    days = week_days(ref_week)
    with st.form(f"planner_{ref_week}"):
        st.write(f"**Wochen-Planer** · Woche ab {week_start(ref_week).strftime('%d.%m.%Y')}")
        rows = []
        for d in days:
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 1.0, 0.7, 0.6, 0.6, 0.8, 0.8])
            with c1:
                title = st.text_input(f"{d.strftime('%a %d.%m.')} Titel", key=f"t_{d}", value="")
            with c2:
                sport = st.selectbox("Sport", SPORTS, index=0, key=f"s_{d}")
            with c3:
                dur = st.number_input("Min", min_value=0, step=5, value=0, key=f"du_{d}")
            with c4:
                rpe = st.number_input("RPE", min_value=0, max_value=10, step=1, value=0, key=f"r_{d}")
            with c5:
                tss = st.number_input("TSS", min_value=0.0, step=5.0, value=0.0, key=f"ts_{d}")
            with c6:
                avg_hr = st.number_input("Avg HR", min_value=0, step=1, value=0, key=f"hr_{d}")
            with c7:
                avg_power = st.number_input("Avg W (Bike)", min_value=0, step=5, value=0, key=f"pw_{d}")
            rows.append((d, title, sport, dur, rpe, tss, avg_hr, avg_power))
        submitted = st.form_submit_button("➕ Woche speichern")
    return submitted, rows

# -------------------- App --------------------
def main():
    ensure_dir()
    settings = load_settings()

    st.title("🏃‍♂️ TrainPeek Pro — Kalender, Phasen & Auto-TSS")

    # --- Einstellungen (FTP/LTHR) ---
    with st.expander("⚙️ Einstellungen (für Auto-TSS)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            ftp = st.number_input("FTP (W, nur Bike)", min_value=0, step=5, value=int(settings.get("ftp_watt") or 0))
        with c2:
            lthr = st.number_input("LTHR / Schwellen-HF (bpm)", min_value=0, step=1, value=int(settings.get("lthr_bpm") or 0))
        if st.button("Einstellungen speichern"):
            settings["ftp_watt"] = int(ftp) if ftp > 0 else None
            settings["lthr_bpm"] = int(lthr) if lthr > 0 else None
            save_settings(settings)
            st.success("Gespeichert. Auto-TSS nutzt diese Werte.")

    # Daten laden
    wdf = load_csv(WORKOUTS_FILE, WORKOUT_COLS)
    pdf = load_csv(PLAN_FILE, PLAN_COLS)

    # Sidebar — Workouts (Einzelerfassung)
    with st.sidebar:
        st.header("Workout hinzufügen")
        d_val = st.date_input("Datum", value=date.today())
        sport = st.selectbox("Sport", SPORTS, index=0)
        title = st.text_input("Titel", value="Workout")
        c1, c2 = st.columns(2)
        with c1:
            dur = st.number_input("Dauer (Min)", 0, step=5, value=60)
            rpe = st.number_input("RPE", 0, 10, 6)
            avg_hr = st.number_input("Durchschn. HF (bpm)", min_value=0, step=1, value=0)
        with c2:
            dist = st.number_input("Distanz (km)", 0.0, step=0.5, value=0.0)
            tss = st.number_input("TSS (falls bekannt)", 0.0, step=5.0, value=0.0)
            avg_power = st.number_input("Durchschn. Watt (Bike)", min_value=0, step=5, value=0)
        notes = st.text_area("Notizen", height=60)

        if st.button("Workout speichern", use_container_width=True):
            row = {
                "date": d_val, "sport": sport, "title": title.strip() or "Workout",
                "duration_min": dur, "distance_km": dist if dist>0 else np.nan,
                "rpe": rpe,
                "tss": tss if tss>0 else np.nan,
                "avg_hr": avg_hr if avg_hr>0 else np.nan,
                "avg_power": avg_power if avg_power>0 else np.nan,
                "notes": notes.strip()
            }
            # Auto-TSS, wenn TSS leer:
            if pd.isna(row["tss"]):
                tss_auto = auto_tss_from_metrics(pd.Series(row), settings)
                if tss_auto is not None:
                    row["tss"] = round(float(tss_auto), 1)
            wdf = pd.concat([wdf, pd.DataFrame([row])], ignore_index=True)
            save_csv(wdf, WORKOUTS_FILE)
            msg = f"Workout gespeichert.{' (TSS automatisch berechnet)' if pd.notna(row['tss']) else ''}"
            st.success(msg)

        st.divider()
        # Sidebar — Phasen & Rennen
        st.header("Planung")
        st.subheader("Phase hinzufügen")
        ph_type = st.selectbox("Phasen-Typ", list(PHASE_TYPES.keys()), index=0)
        ph_title = st.text_input("Titel (optional)", value="")
        pc1, pc2 = st.columns(2)
        with pc1:
            ph_start = st.date_input("Start", value=date.today())
        with pc2:
            ph_end   = st.date_input("Ende",  value=date.today()+timedelta(days=28))
        ph_color = st.color_picker("Farbe", value=PHASE_TYPES[ph_type]["color"])
        ph_notes = st.text_area("Notizen zur Phase", height=60, key="ph_notes")
        if st.button("Phase speichern", use_container_width=True):
            row = pd.DataFrame([{
                "kind":"phase", "phase_type": ph_type,
                "title": (ph_title.strip() or ph_type),
                "start_date": ph_start, "end_date": ph_end,
                "sport":"", "priority":"", "color": ph_color, "notes": ph_notes.strip()
            }])
            pdf = pd.concat([pdf,row], ignore_index=True)
            save_csv(pdf, PLAN_FILE)
            st.success(f"Phase „{ph_type}“ gespeichert.")

        st.subheader("Wettkampf hinzufügen")
        rc1, rc2 = st.columns(2)
        race_title = st.text_input("Eventtitel", value="10k / Marathon / Triathlon")
        race_date  = rc1.date_input("Datum", value=date.today()+timedelta(days=30))
        race_sport = rc2.selectbox("Sport", SPORTS, index=0, key="rsport")
        race_prio  = st.selectbox("Priorität", PRIOS, index=0)
        race_color = st.color_picker("Farbe", value="#F8D7DA", key="rcolor")
        race_notes = st.text_area("Notizen zum Event", height=60, key="rnotes")
        if st.button("Wettkampf speichern", use_container_width=True):
            row = pd.DataFrame([{
                "kind":"race","phase_type":"","title": race_title.strip(),
                "start_date": race_date,"end_date": race_date,
                "sport": race_sport,"priority": race_prio,"color": race_color,"notes": race_notes.strip()
            }])
            pdf = pd.concat([pdf,row], ignore_index=True)
            save_csv(pdf, PLAN_FILE)
            st.success("Wettkampf gespeichert.")

    # ---- Tabs ----
    tab_cal, tab_dash, tab_data = st.tabs(["🗓️ Kalender", "📈 Dashboard", "📤 Daten"])

    # Kalender + Wochen-Planer
    with tab_cal:
        view = st.radio("Ansicht", ["Woche","Monat"], index=0, horizontal=True)
        ref = st.session_state.get("ref_date", date.today())
        c1, c2, c3 = st.columns(3)
        if c1.button("◀︎ Zurück"):
            if view=="Woche":
                ref = ref - timedelta(days=7)
            else:
                ref = (ref.replace(day=1) - timedelta(days=1)).replace(day=1)
        if c2.button("Heute"):
            ref = date.today()
        if c3.button("Vor ▶︎"):
            if view=="Woche":
                ref = ref + timedelta(days=7)
            else:
                end_of_month = (ref.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                ref = (end_of_month + timedelta(days=1)).replace(day=1)
        st.session_state["ref_date"] = ref
        st.markdown("---")

        with st.expander("📝 Wochen-Planer (mit Auto-TSS)", expanded=False):
            submitted, rows = weekly_planner_form(ref)
            if submitted:
                new_rows = []
                for (d, title, sport, dur, rpe, tss, avg_hr, avg_power) in rows:
                    if not title.strip():
                        continue
                    row = {
                        "date": d, "sport": sport, "title": title.strip(),
                        "duration_min": int(dur) if dur else np.nan,
                        "distance_km": np.nan,
                        "rpe": int(rpe) if rpe else np.nan,
                        "tss": float(tss) if tss else np.nan,   # manuell überschreibt Auto-TSS
                        "avg_hr": int(avg_hr) if avg_hr else np.nan,
                        "avg_power": int(avg_power) if avg_power else np.nan,
                        "notes": "planned"
                    }
                    # Auto-TSS nur, wenn tss leer
                    if pd.isna(row["tss"]):
                        tss_auto = auto_tss_from_metrics(pd.Series(row), settings)
                        if tss_auto is not None:
                            row["tss"] = round(float(tss_auto), 1)
                    new_rows.append(row)
                if new_rows:
                    wdf = pd.concat([wdf, pd.DataFrame(new_rows)], ignore_index=True)
                    save_csv(wdf, WORKOUTS_FILE)
                    st.success(f"{len(new_rows)} geplante Workouts gespeichert (Auto-TSS wo möglich).")
                else:
                    st.info("Keine Einträge ausgefüllt.")

        st.markdown("---")

        if view == "Woche":
            st.caption(f"KW {ref.isocalendar()[1]} — Woche ab {week_start(ref).strftime('%d.%m.%Y')}")
            days = week_days(ref)
            cols = st.columns(7)
            for i, d in enumerate(days):
                with cols[i]:
                    st.markdown(f"#### {d.strftime('%a %d.%m.')}")
                    tsd = _ts(d)
                    # Phasen (Timestamp-Vergleiche)
                    ph = pdf[
                        (pdf["kind"]=="phase") &
                        (_ts(pdf["start_date"]) <= tsd) &
                        (_ts(pdf["end_date"])   >= tsd)
                    ]
                    for _, row in ph.iterrows():
                        label = row["title"] or row["phase_type"] or "Phase"
                        bg = row["color"] or PHASE_TYPES.get(row["phase_type"],{}).get("color","#FFFDE7")
                        st.markdown(
                            f"<div style='background:{bg};border:1px solid rgba(0,0,0,.08);border-radius:8px;padding:6px;font-size:12px'><b>{row['phase_type'] or 'Phase'}</b><br/>{label}</div>",
                            unsafe_allow_html=True
                        )
                    # Rennen
                    rc = pdf[
                        (pdf["kind"]=="race") &
                        (_ts(pdf["start_date"]) == tsd)
                    ]
                    for _, r in rc.iterrows():
                        st.markdown(
                            f"<div style='background:{r['color'] or '#F8D7DA'};border:1px solid rgba(0,0,0,.08);border-radius:8px;padding:6px;font-size:12px'><b>🏁 {r['title']}</b><br/>{r['sport']} • Prio {r['priority']}</div>",
                            unsafe_allow_html=True
                        )
                    # Workouts
                    day_w = wdf[_ts(wdf["date"]) == tsd]
                    for _, w in day_w.iterrows():
                        TL = training_load(w, settings)
                        tags = []
                        if pd.notna(w.get("avg_hr")) and w.get("avg_hr")>0:
                            tags.append(f"HR {int(w['avg_hr'])} bpm")
                        if str(w.get("sport","")).lower()=="bike" and pd.notna(w.get("avg_power")) and w.get("avg_power")>0:
                            tags.append(f"{int(w['avg_power'])} W")
                        if str(w.get("notes","")).lower()=="planned":
                            tags.append("geplant")
                        tag_str = (" · " + " | ".join(tags)) if tags else ""
                        st.markdown(
                            f"<div style='border:1px solid rgba(0,0,0,.08);border-radius:8px;padding:6px;font-size:12px'><b>{w['title']}</b>{tag_str}<br/>{w['sport']} • {int(w.get('duration_min') or 0)} min • TL {int(round(TL))}</div>",
                            unsafe_allow_html=True
                        )
        else:
            st.caption(ref.strftime("%B %Y"))
            first = ref.replace(day=1)
            start = first - timedelta(days=first.weekday())
            cur = start
            for _ in range(6):
                cols = st.columns(7)
                for i in range(7):
                    d = cur
                    with cols[i]:
                        muted = (d.month != ref.month)
                        style = "opacity:.5" if muted else ""
                        st.markdown(f"<div style='{style}'><b>{d.day}</b></div>", unsafe_allow_html=True)
                        tsd = _ts(d)
                        # Phasen
                        ph = pdf[
                            (pdf["kind"]=="phase") &
                            (_ts(pdf["start_date"]) <= tsd) &
                            (_ts(pdf["end_date"])   >= tsd)
                        ]
                        if not ph.empty:
                            ptype = ph.iloc[0]["phase_type"] or "Phase"
                            st.markdown(f"<div style='font-size:11px;border:1px dashed rgba(0,0,0,.2);border-radius:6px;padding:3px'>{ptype}</div>", unsafe_allow_html=True)
                        # Rennen
                        rc = pdf[
                            (pdf["kind"]=="race") &
                            (_ts(pdf["start_date"]) == tsd)
                        ]
                        for _, r in rc.iterrows():
                            st.markdown(f"<div style='font-size:11px;border:1px solid rgba(0,0,0,.1);border-radius:6px;padding:3px'>🏁 {r['title']}</div>", unsafe_allow_html=True)
                        # Workouts
                        cnt = int((_ts(wdf["date"]) == tsd).sum())
                        if cnt:
                            st.markdown(f"<div style='font-size:11px'>Workouts: {int(cnt)}</div>", unsafe_allow_html=True)
                    cur += timedelta(days=1)

    # Dashboard
    with tab_dash:
        wk = weekly_summary(wdf, settings)
        if wk.empty:
            st.info("Noch keine Workouts.")
        else:
            st.subheader("Wöchentliche Trainingslast")
            st.dataframe(wk, use_container_width=True)
            fig1, ax1 = plt.subplots()
            ax1.bar(wk["week_start"].astype(str), wk["TL"])
            ax1.set_title("Wöchentliche Trainingslast (TL)")
            ax1.set_xlabel("Woche (Montag)")
            ax1.set_ylabel("TL (TSS oder Dauer×RPE)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig1, clear_figure=True)

        st.subheader("Performance Management Chart (ATL/CTL/TSB)")
        daily = daily_load(wdf, settings)
        pmc = compute_pmc(daily)
        if pmc.empty:
            st.info("Noch keine Daten für PMC.")
        else:
            fig2, ax2 = plt.subplots()
            ax2.plot(pmc["date"], pmc["CTL"], label="CTL (42d)")
            ax2.plot(pmc["date"], pmc["ATL"], label="ATL (7d)")
            ax2.set_xlabel("Datum"); ax2.set_ylabel("Belastung"); ax2.set_title("ATL & CTL")
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

            fig3, ax3 = plt.subplots()
            ax3.plot(pmc["date"], pmc["TSB"]); ax3.axhline(0)
            ax3.set_xlabel("Datum"); ax3.set_ylabel("TSB"); ax3.set_title("TSB (Frische)")
            st.pyplot(fig3, clear_figure=True)

        st.caption("Auto-TSS nutzt deine FTP/LTHR-Einstellungen. Falls beides fehlt, wird Dauer×RPE verwendet.")

    # Daten
    with tab_data:
        st.subheader("Workouts")
        st.dataframe(wdf if not wdf.empty else pd.DataFrame(columns=WORKOUT_COLS), use_container_width=True)
        st.download_button("📥 Workouts.csv", (wdf if not wdf.empty else pd.DataFrame(columns=WORKOUT_COLS)).to_csv(index=False).encode("utf-8"), "workouts.csv")
        st.divider()
        st.subheader("Plan (Phasen & Wettkämpfe)")
        st.dataframe(pdf if not wdf.empty else pd.DataFrame(columns=PLAN_COLS), use_container_width=True)
        st.download_button("📥 Plan.csv", (pdf if not pdf.empty else pd.DataFrame(columns=PLAN_COLS)).to_csv(index=False).encode("utf-8"), "plan.csv")
        st.caption("Hinweis: Auf Streamlit Cloud sind Dateien nicht dauerhaft. Für Persistenz später Supabase/Postgres nutzen.")

if __name__ == "__main__":
    main()
