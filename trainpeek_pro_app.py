# trainpeek_pro_app.py
# TrainPeek Pro — Neon Dark App
# - Linke Seiten-Navigation: Startseite, Planer (Kalender), Analysen, Einstellungen
# - Startseite: KPI-Karten (TSS Woche, Form/TSB, ATL, CTL) + Wochen-Übersicht (geplant/erledigt + nächste Workouts)
# - Runde, saubere Karten (größere Rundungen)
# - Bestehendes: Phasen, Auto-TSS, Status planned/done/skipped, Forecast, Neon-Plotly Charts

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta, datetime
import json, uuid
import plotly.graph_objects as go

st.set_page_config(page_title="TrainPeek Pro — Neon", page_icon="🏃‍♂️", layout="wide")

DATA_DIR = Path("data")
WORKOUTS_FILE = DATA_DIR / "workouts.csv"
PLAN_FILE = DATA_DIR / "plan.csv"
SETTINGS_FILE = DATA_DIR / "settings.json"

# Workouts inkl. ID/Status/HR/Power (Auto-TSS)
WORKOUT_COLS = [
    "id","date","sport","title","duration_min","distance_km","rpe","tss",
    "avg_hr","avg_power","status","notes"
]
PLAN_COLS = [
    "kind","phase_type","title","start_date","end_date","sport","priority","color","notes"
]  # kind: phase|race

SPORTS = ["Run","Bike","Swim","Strength","Other"]
PRIOS  = ["A","B","C"]

PHASE_TYPES = {
    "Grundlage": {"color": "#133253"},
    "Aufbau":    {"color": "#173b2b"},
    "Spitze":    {"color": "#4a2e12"},
    "Taper":     {"color": "#2b1947"},
    "Erholung":  {"color": "#4a1f2a"},
}

# -------------------- NEON THEME --------------------
NEON = {
    "bg":      "#0E1117",
    "card":    "#161B22",
    "muted":   "#9CA3AF",
    "text":    "#E5E7EB",
    "primary": "#60A5FA",
    "cyan":    "#22D3EE",
    "pink":    "#F472B6",
    "violet":  "#A78BFA",
    "amber":   "#F59E0B",
    "green":   "#10B981",
    "orange":  "#FB923C",
}

def inject_theme():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --bg: {NEON['bg']};
  --card: {NEON['card']};
  --text: {NEON['text']};
  --muted: {NEON['muted']};
  --primary: {NEON['primary']};
  --cyan: {NEON['cyan']};
  --pink: {NEON['pink']};
  --violet: {NEON['violet']};
  --amber: {NEON['amber']};
  --green: {NEON['green']};
}}
html, body, [class*="css"] {{
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}}
/* App-Shell */
.tp-app {{
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 16px;
}}
/* Sidebar */
.tp-side {{
  position: sticky; top: 16px;
  height: calc(100vh - 32px);
  background: var(--card);
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 18px;                 /* -> runder */
  padding: 14px;
  box-shadow: 0 1px 1px rgba(0,0,0,.25), 0 10px 24px rgba(0,0,0,.35);
}}
.tp-side a {{
  display:block; padding:10px 12px; border-radius: 12px; color: var(--text); text-decoration: none;
  border: 1px solid transparent;
}}
.tp-side a:hover {{ background: rgba(255,255,255,.04); border-color: rgba(255,255,255,.08); }}
.tp-side .active {{ background: rgba(96,165,250,.12); border-color: rgba(96,165,250,.35); color: #dbeafe; }}
/* Karten */
.tp-card {{
  background: var(--card);
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 18px;                 /* -> runder */
  padding: 14px;
  box-shadow: 0 1px 1px rgba(0,0,0,.25), 0 10px 24px rgba(0,0,0,.35);
}}
/* KPI */
.tp-kpi {{
  background: var(--card);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px;
  padding: 14px;
}}
.tp-kpi .label {{ color: var(--muted); font-size: 12px; }}
.tp-kpi .value {{ font-size: 28px; font-weight: 700; }}
/* Chips / Status */
.tp-chip {{
  display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600;
}}
.tp-chip--planned {{ color: var(--amber); border:1px solid var(--amber); background: rgba(245,158,11,.08); }}
.tp-chip--done    {{ color: #052e1a; background: var(--green); }}
.tp-chip--skip    {{ color: #111827; background: #9CA3AF; }}
/* Button klein */
.tp-btn-icon {{
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 12px; padding:6px 10px; font-weight:600;
}}
.tp-btn-icon:hover {{ border-color: var(--primary); box-shadow: 0 0 0 2px rgba(96,165,250,.2) inset; }}
/* Tageskopf im Kalender */
.tp-dayhead {{ font-size:13px; font-weight:700; color:#cbd5e1; margin-bottom:6px; }}
/* Tabellen / Plotly */
[data-testid="stDataFrame"] div, .plotly .main-svg {{ color: var(--text) !important; }}
/* Content-Container rechts */
.tp-main {{ }}
/* Headline */
.tp-h1 {{ font-size: 20px; font-weight: 700; margin: 0 0 8px 0; }}
.tp-sub {{ color: var(--muted); font-size: 13px; }}
</style>
        """,
        unsafe_allow_html=True,
    )

# ---------- Timestamp Helper ----------
def _ts(x):
    if isinstance(x, pd.Series): return pd.to_datetime(x, errors="coerce").dt.normalize()
    return pd.Timestamp(x).normalize()

# -------------------- Storage helpers --------------------
def ensure_dir(): DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_settings():
    ensure_dir()
    base = {"ftp_watt": None, "lthr_bpm": None, "strava_embed_url": "", "forecast_days": 28}
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            base.update({k: data.get(k, base[k]) for k in base})
        except Exception:
            pass
    return base

def save_settings(s: dict):
    ensure_dir()
    SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists(): df = pd.read_csv(path)
    else: df = pd.DataFrame(columns=columns)

    for c in columns:
        if c not in df.columns: df[c] = np.nan
    df = df[columns].copy()

    df = df.replace(r'^\s*$', np.nan, regex=True)

    for c in ["duration_min","distance_km","rpe","tss","avg_hr","avg_power"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["id","title","sport","priority","notes","kind","color","phase_type","status"]:
        if c in df.columns: df[c] = df[c].astype(str).fillna("")

    for c in ["date","start_date","end_date"]:
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.date

    if {"start_date","end_date"}.issubset(df.columns):
        df = df[df["start_date"].notna() & df["end_date"].notna()].copy()

    if "id" in df.columns:
        mask_id = df["id"].astype(str).str.strip().eq("") | df["id"].isna()
        if mask_id.any():
            df.loc[mask_id, "id"] = [str(uuid.uuid4()) for _ in range(int(mask_id.sum()))]

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
        df.loc[df["status"].isin(["nan","none"]), "status"] = ""
        notes_lower = df["notes"].astype(str).str.lower() if "notes" in df.columns else pd.Series("", index=df.index)
        default_status = pd.Series(np.where(notes_lower.eq("planned"), "planned", "done"), index=df.index)
        mask_status = df["status"].isna() | (df["status"].str.strip() == "")
        df.loc[mask_status, "status"] = default_status.loc[mask_status]
        df.loc[~df["status"].isin(["planned","done","skipped"]), "status"] = "done"

    return df

def save_csv(df: pd.DataFrame, path: Path):
    ensure_dir(); df.to_csv(path, index=False)

# -------------------- Load / TSS / Metrics --------------------
def auto_tss_from_metrics(row: pd.Series, settings: dict) -> float | None:
    dur_min = float(row.get("duration_min") or 0.0)
    if dur_min <= 0: return None
    dur_h = dur_min / 60.0
    if str(row.get("sport","")).lower() == "bike":
        avg_power = row.get("avg_power"); ftp = settings.get("ftp_watt")
        if pd.notna(avg_power) and avg_power and ftp and ftp > 0:
            IF = float(avg_power) / float(ftp); return dur_h * (IF**2) * 100.0
    avg_hr = row.get("avg_hr"); lthr = settings.get("lthr_bpm")
    if pd.notna(avg_hr) and avg_hr and lthr and lthr > 0:
        HRr = float(avg_hr) / float(lthr); return dur_h * (HRr**2) * 100.0
    return None

def training_load(row: pd.Series, settings: dict) -> float:
    tss = row.get("tss", np.nan)
    if pd.notna(tss) and float(tss) > 0: return float(tss)
    tss_auto = auto_tss_from_metrics(row, settings)
    if tss_auto is not None: return float(tss_auto)
    return float(row.get("duration_min") or 0) * float(row.get("rpe") or 0)

def daily_load(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["date","TL"])
    tmp = df.copy()
    tmp["TL"] = tmp.apply(lambda r: training_load(r, settings), axis=1)
    return tmp.groupby("date", as_index=False)["TL"].sum().sort_values("date")

def exp_ema(values: np.ndarray, tau_days: float) -> np.ndarray:
    alpha = 1 - np.exp(-1 / tau_days)
    ema, prev = [], 0.0
    for x in values: prev = prev + (x - prev) * alpha; ema.append(prev)
    return np.array(ema, dtype=float)

def compute_pmc(daily: pd.DataFrame, tau_atl=7.0, tau_ctl=42.0) -> pd.DataFrame:
    if daily.empty: return pd.DataFrame(columns=["date","TL","ATL","CTL","TSB"])
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    frame = pd.DataFrame({"date": dates})
    merged = frame.merge(daily, on="date", how="left").fillna({"TL": 0.0})
    ATL = exp_ema(merged["TL"].to_numpy(float), tau_atl)
    CTL = exp_ema(merged["TL"].to_numpy(float), tau_ctl)
    merged["ATL"] = ATL; merged["CTL"] = CTL; merged["TSB"] = CTL - ATL
    return merged

def compute_pmc_forecast(actual_daily: pd.DataFrame, planned_daily: pd.DataFrame, horizon_days: int = 28):
    today = date.today()
    act = actual_daily.copy()
    if not act.empty: act = act[act["date"] <= today]
    pmc_actual = compute_pmc(act)

    fore = act.copy()
    if not planned_daily.empty: fore = pd.concat([fore, planned_daily[planned_daily["date"] > today]], ignore_index=True)
    start = min([x for x in [fore["date"].min() if not fore.empty else today, today] if x is not None])
    end = today + timedelta(days=horizon_days)
    all_days = pd.DataFrame({"date": pd.date_range(start, end, freq="D").date})
    merged = all_days.merge(fore, on="date", how="left").fillna({"TL": 0.0})
    pmc_fore = compute_pmc(merged)
    return pmc_actual, pmc_fore

def week_start(d: date) -> date: return d - timedelta(days=d.weekday())
def week_days(ref: date): ws = week_start(ref); return [ws + timedelta(days=i) for i in range(7)]

def weekly_summary(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["week_start","sessions","duration_min","distance_km","TL"])
    tmp = df.copy()
    tmp["week_start"] = tmp["date"].apply(week_start)
    tmp["TL"] = tmp.apply(lambda r: training_load(r, settings), axis=1)
    return tmp.groupby("week_start", as_index=False).agg(
        sessions=("date","count"),
        duration_min=("duration_min","sum"),
        distance_km=("distance_km","sum"),
        TL=("TL","sum"),
    ).sort_values("week_start")

# -------------------- Charts (dark) --------------------
def fig_layout_dark(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=NEON["card"],
        plot_bgcolor=NEON["card"],
        font=dict(family="Inter", size=13, color=NEON["text"]),
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text=title, x=0.01, xanchor="left") if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0.01),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)")
    return fig

def neon_line(x, ys: dict, title=None):
    fig = go.Figure()
    for name, series in ys.items():
        color = series.get("color"); y = series["y"]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=3, color=color), name=name))
    return fig_layout_dark(fig, title)

def neon_bar(x, y, title=None, color=None):
    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=color or NEON["primary"], line=dict(color="rgba(255,255,255,.15)", width=1))))
    return fig_layout_dark(fig, title)

def tsb_gauge(value, title="TSB (Form)"):
    vmax = 40; vmin = -40
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title},
        gauge={
            "axis": {"range":[vmin, vmax], "tickcolor": NEON["muted"]},
            "bar": {"color": NEON["cyan"]},
            "bgcolor": NEON["card"],
            "borderwidth": 1, "bordercolor": "rgba(255,255,255,.1)",
            "steps": [
                {"range":[vmin,-10],"color":"rgba(244,114,182,.25)"},
                {"range":[-10,10],"color":"rgba(148,163,184,.2)"},
                {"range":[10,vmax],"color":"rgba(16,185,129,.25)"},
            ],
        }
    ))
    return fig_layout_dark(fig)

# Heatmap: robuste Version (fix für Duplikate)
def year_heatmap(daily_df: pd.DataFrame, year: int):
    all_days = pd.DataFrame({"date": pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")})
    if daily_df.empty:
        frame = all_days.copy(); frame["TL"] = 0.0
    else:
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["date"].dt.year == year][["date", "TL"]]
        frame = all_days.merge(df, on="date", how="left").fillna({"TL": 0.0})

    iso = frame["date"].dt.isocalendar()
    frame["dow"] = frame["date"].dt.dayofweek.astype(int)
    frame["week"] = iso.week.astype(int)
    jan_mask = frame["date"].dt.month.eq(1) & (frame["week"] > 50)
    frame.loc[jan_mask, "week"] = 0

    agg = frame.groupby(["dow", "week"], as_index=False)["TL"].sum()
    weeks_sorted = sorted(agg["week"].unique())

    pivot = agg.pivot(index="dow", columns="week", values="TL").reindex(columns=weeks_sorted)
    pivot = pivot.reindex(index=[0,1,2,3,4,5,6])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=["Mo","Di","Mi","Do","Fr","Sa","So"],
        colorscale=[[0,"#0ea5e9"], [0.5,"#a78bfa"], [1,"#f472b6"]],
        colorbar=dict(title="TL"),
        hovertemplate="Woche %{x} • %{y}<br>TL=%{z:.0f}<extra></extra>",
    ))
    return fig_layout_dark(fig, f"Jahres-Heatmap {year}")

# -------------------- UI helpers --------------------
def status_styles(status: str):
    status = (status or "").lower()
    if status == "planned": return "border:1px solid var(--amber); background: rgba(245,158,11,.08)"
    if status == "done":    return "border:1px solid #34d399; background: rgba(16,185,129,.15)"
    if status == "skipped": return "border:1px solid #9CA3AF; background: rgba(156,163,175,.15)"
    return "border:1px solid rgba(255,255,255,.08); background: var(--card)"

def weekly_planner_form(ref_week: date):
    days = week_days(ref_week)
    with st.form(f"planner_{ref_week}"):
        st.write(f"**Wochen-Planer** · Woche ab {week_start(ref_week).strftime('%d.%m.%Y')}")
        rows = []
        for d in days:
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1.4, 1.0, 0.8, 0.8, 0.8, 0.9, 0.9])
            with c1: title = st.text_input(f"{d.strftime('%a %d.%m.')} Titel", key=f"t_{d}", value="")
            with c2: sport = st.selectbox("Sport", SPORTS, index=0, key=f"s_{d}")
            with c3: dur = st.number_input("Min", min_value=0, step=5, value=0, key=f"du_{d}")
            with c4: rpe = st.number_input("RPE", min_value=0, max_value=10, step=1, value=0, key=f"r_{d}")
            with c5: tss = st.number_input("TSS", min_value=0.0, step=5.0, value=0.0, key=f"ts_{d}")
            with c6: avg_hr = st.number_input("Avg HR", min_value=0, step=1, value=0, key=f"hr_{d}")
            with c7: avg_power = st.number_input("Avg W (Bike)", min_value=0, step=5, value=0, key=f"pw_{d}")
            rows.append((d, title, sport, dur, rpe, tss, avg_hr, avg_power))
        submitted = st.form_submit_button("➕ Woche speichern")
    return submitted, rows

def mark_done_ui(row_id: str):
    return st.button("✅ Erledigt", key=f"done_{row_id}")

# ===================== APP =====================
def main():
    inject_theme()
    ensure_dir()
    settings = load_settings()

    # ----- App-Shell mit linker Navigation -----
    st.markdown("<div class='tp-app'>", unsafe_allow_html=True)

    # Sidebar (eigene Navigation)
    with st.container():
        st.markdown("<div class='tp-side'>", unsafe_allow_html=True)
        st.markdown("<div class='tp-h1'>🏃‍♂️ TrainPeek Pro</div><div class='tp-sub'>Neon Dark</div><hr/>", unsafe_allow_html=True)

        # Navigation via Radio (wir stylen Links dazu)
        nav = st.radio("Navigation", ["Startseite", "Planer (Kalender)", "Analysen", "Einstellungen"], index=0, label_visibility="collapsed")
        # Fake-Links (optisch), aktuelle als .active markieren
        labels = ["Startseite", "Planer (Kalender)", "Analysen", "Einstellungen"]
        html = "<div>"
        for lab in labels:
            cls = "active" if lab == nav else ""
            html += f"<a class='{cls}'>{lab}</a>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # /tp-side

    # Main Content
    st.markdown("<div class='tp-main'>", unsafe_allow_html=True)

    # ------------- Daten laden -------------
    wdf = load_csv(WORKOUTS_FILE, WORKOUT_COLS)
    pdf = load_csv(PLAN_FILE, PLAN_COLS)

    # ====== STARTSEITE ======
    if nav == "Startseite":
        st.markdown("<div class='tp-card'><div class='tp-h1'>Startseite</div><div class='tp-sub'>Schneller Überblick</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        today = date.today()
        this_week_start = week_start(today)
        this_week_end = this_week_start + timedelta(days=6)

        # TL (Woche)
        week_df = wdf[(wdf["date"] >= this_week_start) & (wdf["date"] <= this_week_end)]
        week_daily = daily_load(week_df, settings)
        week_tl = float(week_daily["TL"].sum()) if not week_daily.empty else 0.0

        # PMC heute (ATL/CTL/TSB)
        all_daily = daily_load(wdf, settings)
        pmc = compute_pmc(all_daily)
        atl = ctl = tsb = 0.0
        if not pmc.empty:
            row = pmc[pmc["date"] == today]
            if row.empty: row = pmc.iloc[-1:]
            atl = float(row["ATL"].iloc[0]); ctl = float(row["CTL"].iloc[0]); tsb = float(row["TSB"].iloc[0])

        # KPI-Karten
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='tp-kpi'><div class='label'>Wochen-TSS</div><div class='value'>{int(round(week_tl))}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='tp-kpi'><div class='label'>Form (TSB) heute</div><div class='value'>{tsb:.0f}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='tp-kpi'><div class='label'>ATL</div><div class='value'>{atl:.0f}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='tp-kpi'><div class='label'>CTL</div><div class='value'>{ctl:.0f}</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Wochen-Übersicht: geplant vs erledigt + kommende geplante
        st.markdown("<div class='tp-card'><b>Diese Woche</b></div>", unsafe_allow_html=True)
        week_w = wdf[(wdf["date"] >= this_week_start) & (wdf["date"] <= this_week_end)]
        if week_w.empty:
            st.caption("Noch keine Workouts in dieser Woche.")
        else:
            n_planned = (week_w["status"].str.lower()=="planned").sum()
            n_done = (week_w["status"].str.lower()=="done").sum()
            st.markdown(
                f"<div class='tp-card' style='margin-top:8px'>"
                f"<div style='display:flex;gap:12px;flex-wrap:wrap'>"
                f"<div class='tp-chip tp-chip--planned'>geplant: {int(n_planned)}</div>"
                f"<div class='tp-chip tp-chip--done'>erledigt: {int(n_done)}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            # Nächste geplante (heute..Ende Woche)
            upcoming = week_w[(week_w["status"].str.lower()=="planned") & (_ts(week_w["date"]) >= _ts(today))]
            if not upcoming.empty:
                st.markdown("<div class='tp-card' style='margin-top:8px'><b>Nächste geplante Einheiten</b></div>", unsafe_allow_html=True)
                for _, w in upcoming.sort_values("date").iterrows():
                    tl = int(round(training_load(w, settings)))
                    st.markdown(
                        f"<div class='tp-card' style='{status_styles('planned')}; margin-top:6px'>"
                        f"<b>{w['title']}</b> — {w['sport']} • {int(float(w.get('duration_min') or 0))} min"
                        f"<br><span style='color:{NEON['cyan']};font-weight:700'>TL {tl}</span>"
                        f"<div style='color:var(--muted); font-size:12px; margin-top:2px'>{pd.to_datetime(w['date']).strftime('%a, %d.%m.')}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    # ====== PLANER (Kalender) ======
    if nav == "Planer (Kalender)":
        st.markdown("<div class='tp-card'><div class='tp-h1'>Planer</div><div class='tp-sub'>Kalender & Wochen-Planer</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Navigation Woche/Monat
        view = st.radio("Ansicht", ["Woche","Monat"], index=0, horizontal=True)
        ref = st.session_state.get("ref_date", date.today())
        c1, c2, c3 = st.columns(3)
        if c1.button("◀︎ Zurück"):
            ref = ref - timedelta(days=7) if view=="Woche" else (ref.replace(day=1) - timedelta(days=1)).replace(day=1)
        if c2.button("Heute"): ref = date.today()
        if c3.button("Vor ▶︎"):
            if view=="Woche": ref = ref + timedelta(days=7)
            else:
                end_of_month = (ref.replace(day=28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
                ref = (end_of_month + timedelta(days=1)).replace(day=1)
        st.session_state["ref_date"] = ref
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Wochen-Planer
        with st.expander("📝 Wochen-Planer (Auto-TSS)"):
            submitted, rows = weekly_planner_form(ref)
            if submitted:
                new_rows = []
                for (d, title, sport, dur, rpe, tss, avg_hr, avg_power) in rows:
                    if not title.strip(): continue
                    row = {
                        "id": str(uuid.uuid4()), "date": d, "sport": sport, "title": title.strip(),
                        "duration_min": int(dur) if dur else np.nan, "distance_km": np.nan,
                        "rpe": int(rpe) if rpe else np.nan, "tss": float(tss) if tss else np.nan,
                        "avg_hr": int(avg_hr) if avg_hr else np.nan, "avg_power": int(avg_power) if avg_power else np.nan,
                        "status": "planned", "notes": "planned"
                    }
                    if pd.isna(row["tss"]):
                        tss_auto = auto_tss_from_metrics(pd.Series(row), settings)
                        if tss_auto is not None: row["tss"] = round(float(tss_auto), 1)
                    new_rows.append(row)
                if new_rows:
                    wdf = pd.concat([wdf, pd.DataFrame(new_rows)], ignore_index=True); save_csv(wdf, WORKOUTS_FILE)
                    st.success(f"{len(new_rows)} geplante Workouts gespeichert.")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Anzeige
        if view == "Woche":
            st.caption(f"KW {ref.isocalendar()[1]} — Woche ab {week_start(ref).strftime('%d.%m.%Y')}")
            days = week_days(ref); cols = st.columns(7)
            for i, d in enumerate(days):
                with cols[i]:
                    st.markdown(f"<div class='tp-dayhead'>{d.strftime('%a %d.%m.')}</div>", unsafe_allow_html=True)
                    tsd = _ts(d)
                    # Phasen
                    ph = pdf[(pdf["kind"]=="phase") & (_ts(pdf["start_date"])<=tsd) & (_ts(pdf["end_date"])>=tsd)]
                    for _, row in ph.iterrows():
                        label = row["title"] or row["phase_type"] or "Phase"
                        bg = PHASE_TYPES.get(row["phase_type"],{}).get("color", "#1f2937")
                        st.markdown(f"<div class='tp-card' style='background:{bg};border-color:rgba(255,255,255,.15)'><b>{row['phase_type'] or 'Phase'}</b><br/>{label}</div>", unsafe_allow_html=True)
                    # Rennen
                    rc = pdf[(pdf["kind"]=="race") & (_ts(pdf["start_date"]) == tsd)]
                    for _, r in rc.iterrows():
                        st.markdown(f"<div class='tp-card' style='border-color:rgba(255,255,255,.2)'><b>🏁 {r['title']}</b><br/>{r['sport']} • Prio {r['priority']}</div>", unsafe_allow_html=True)
                    # Workouts
                    day_w = wdf[_ts(wdf["date"]) == tsd]
                    for _, w in day_w.iterrows():
                        TL = training_load(w, settings)
                        chip = "tp-chip tp-chip--planned" if w.get("status","")== "planned" else "tp-chip tp-chip--done" if w.get("status","")=="done" else "tp-chip tp-chip--skip"
                        st.markdown(
                            f"<div class='tp-card' style='{status_styles(w.get('status',''))}'>"
                            f"<div style='display:flex;justify-content:space-between;gap:8px;align-items:center'>"
                            f"<div><b>{w['title']}</b><br/><span style='color:var(--muted)'>{w['sport']} • {int(float(w.get('duration_min') or 0))} min</span></div>"
                            f"<div class='{chip}'> {w.get('status','')} </div>"
                            f"</div>"
                            f"<div style='margin-top:6px;color:{NEON['cyan']};font-weight:700'>TL {int(round(TL))}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        if w.get("status","") != "done":
                            if mark_done_ui(w["id"]):
                                wdf.loc[wdf["id"]==w["id"], "status"] = "done"; save_csv(wdf, WORKOUTS_FILE); st.experimental_rerun()
        else:
            st.caption(ref.strftime("%B %Y"))
            first = ref.replace(day=1); start = first - timedelta(days=first.weekday()); cur = start
            for _ in range(6):
                cols = st.columns(7)
                for i in range(7):
                    d = cur; tsd = _ts(d)
                    with cols[i]:
                        muted = (d.month != ref.month)
                        st.markdown(f"<div class='tp-dayhead' style='opacity:{.5 if muted else 1}'><b>{d.day}</b></div>", unsafe_allow_html=True)
                        ph = pdf[(pdf["kind"]=="phase") & (_ts(pdf["start_date"])<=tsd) & (_ts(pdf["end_date"])>=tsd)]
                        if not ph.empty:
                            ptype = ph.iloc[0]["phase_type"] or "Phase"
                            st.markdown(f"<div class='tp-card' style='padding:4px 8px;font-size:12px'>{ptype}</div>", unsafe_allow_html=True)
                        rc = pdf[(pdf["kind"]=="race") & (_ts(pdf["start_date"]) == tsd)]
                        for _, r in rc.iterrows():
                            st.markdown(f"<div class='tp-card' style='padding:4px 8px;font-size:12px'>🏁 {r['title']}</div>", unsafe_allow_html=True)
                        day_w = wdf[_ts(wdf["date"]) == tsd]
                        if not day_w.empty:
                            cnt = len(day_w); n_done = (day_w["status"].str.lower()=="done").sum()
                            st.markdown(f"<div style='color:#9CA3AF;font-size:12px'>Workouts: {cnt} (✅ {n_done})</div>", unsafe_allow_html=True)
                    cur += timedelta(days=1)

    # ====== ANALYSEN ======
    if nav == "Analysen":
        st.markdown("<div class='tp-card'><div class='tp-h1'>Analysen</div><div class='tp-sub'>Weekly Load, PMC & Heatmap</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Zeitraum-Switcher
        win = st.radio("Zeitraum", ["1W","1M","3M","1J"], index=2, horizontal=True)
        today = date.today()
        ranges = {"1W":7, "1M":30, "3M":90, "1J":365}
        start_win = today - timedelta(days=ranges[win])

        actual_df = wdf[(wdf["status"].str.lower()=="done") | (_ts(wdf["date"]) <= _ts(today))]
        planned_df = wdf[(wdf["status"].str.lower()=="planned") & (_ts(wdf["date"]) >= _ts(today))]
        actual_daily = daily_load(actual_df, settings)
        planned_daily = daily_load(planned_df, settings)

        wk = weekly_summary(wdf, settings)
        if not wk.empty:
            wkf = wk[wk["week_start"] >= start_win]
            st.plotly_chart(
                neon_bar(wkf["week_start"].astype(str), wkf["TL"], title="Wöchentliche Trainingslast (TL)", color=NEON["violet"]),
                use_container_width=True
            )
        else:
            st.info("Noch keine Workouts für Weekly Load.")

        pmc_actual, pmc_fore = compute_pmc_forecast(actual_daily, planned_daily, horizon_days=int(settings.get("forecast_days") or 28))
        if pmc_actual.empty and pmc_fore.empty:
            st.info("Keine Daten für PMC.")
        else:
            if not pmc_actual.empty: pmc_actual = pmc_actual[pmc_actual["date"] >= start_win]
            if not pmc_fore.empty:   pmc_fore   = pmc_fore[pmc_fore["date"] >= start_win]
            xA = pmc_actual["date"] if not pmc_actual.empty else []
            fig_lines = neon_line(
                xA if len(xA)>0 else (pmc_fore["date"] if not pmc_fore.empty else []),
                {
                    "CTL (Ist)": {"y": pmc_actual["CTL"] if not pmc_actual.empty else [], "color": NEON["cyan"]},
                    "ATL (Ist)": {"y": pmc_actual["ATL"] if not pmc_actual.empty else [], "color": NEON["pink"]},
                },
                title="ATL / CTL — Ist"
            )
            if not pmc_fore.empty:
                fig_lines.add_trace(go.Scatter(x=pmc_fore["date"], y=pmc_fore["CTL"], mode="lines", name="CTL (Forecast)", line=dict(width=3, dash="dash", color=NEON["cyan"])))
                fig_lines.add_trace(go.Scatter(x=pmc_fore["date"], y=pmc_fore["ATL"], mode="lines", name="ATL (Forecast)", line=dict(width=3, dash="dash", color=NEON["pink"])))
                fig_layout_dark(fig_lines)
            st.plotly_chart(fig_lines, use_container_width=True)

            # TSB heute
            tsb_today = None
            if not pmc_fore.empty:
                row = pmc_fore[pmc_fore["date"]==today]
                if not row.empty: tsb_today = float(row["TSB"].iloc[0])
            elif not pmc_actual.empty:
                row = pmc_actual.iloc[-1:]; tsb_today = float(row["TSB"].iloc[0])
            if tsb_today is not None:
                st.plotly_chart(tsb_gauge(tsb_today, title="TSB (Form) heute"), use_container_width=True)

        # Jahres-Heatmap
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        year = st.slider("Jahr", min_value=2018, max_value=date.today().year, value=date.today().year, step=1)
        daily_all = daily_load(wdf, settings)
        st.plotly_chart(year_heatmap(daily_all, year), use_container_width=True)

    # ====== EINSTELLUNGEN ======
    if nav == "Einstellungen":
        st.markdown("<div class='tp-card'><div class='tp-h1'>Einstellungen</div><div class='tp-sub'>Ziele & Integrationen</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            ftp = st.number_input("FTP (W, nur Bike)", min_value=0, step=5, value=int(settings.get("ftp_watt") or 0))
        with c2:
            lthr = st.number_input("LTHR (bpm)", min_value=0, step=1, value=int(settings.get("lthr_bpm") or 0))
        with c3:
            forecast_days = st.number_input("Forecast (Tage)", min_value=7, max_value=90, step=7, value=int(settings.get("forecast_days") or 28))
        if st.button("Speichern"):
            settings["ftp_watt"] = int(ftp) if ftp>0 else None
            settings["lthr_bpm"] = int(lthr) if lthr>0 else None
            settings["forecast_days"] = int(forecast_days)
            save_settings(settings)
            st.success("Einstellungen gespeichert.")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.caption("Tipp: Für echte Persistenz (Login, Cloud) später Supabase einbauen. Für Strava-Sync brauchen wir OAuth – kann ich dir nachziehen.")

    # ----- Shell schließen -----
    st.markdown("</div>", unsafe_allow_html=True)   # /tp-main
    st.markdown("</div>", unsafe_allow_html=True)   # /tp-app

if __name__ == "__main__":
    main()

