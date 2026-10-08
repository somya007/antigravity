"""
═══════════════════════════════════════════════════════════════════════════
  Adaptive Error Control System — Interactive Streamlit Dashboard
═══════════════════════════════════════════════════════════════════════════
  Clean light-themed interface with traffic-light status, reliability
  gauge, and enhanced ORBGRAND correction engine (30K queries, weight 5).

  Run:  streamlit run streamlit_dashboard.py
═══════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, random, warnings, math
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")

from core.polynomials import get_polynomial
from core.crc_encoder import CRCEncoder
from core.crc_decoder import CRCDecoder
from channel.channel_model import ChannelModel
from decoding.bp_decoder import BPDecoder
from decoding.orbgrand_decoder import ORBGRANDDecoder
from evaluation.checksum_comparison import InternetChecksum
from controller.intelligent_controller import (
    ChannelStateMonitor, AliasingPredictor,
    ALIASING_THRESHOLD, ORBGRAND_NORMAL, ORBGRAND_AGGRESSIVE, CHANNEL_WINDOW_SIZE,
)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Adaptive Error Control System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  LIGHT-THEME CSS
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
.stApp {
    background: #FAFBFC !important;
    color: #111827;
    font-family: 'Inter', -apple-system, sans-serif;
}
section[data-testid="stSidebar"] {
    background: #F3F4F6 !important;
    border-right: 1px solid #E5E7EB;
}
footer, .stDeployButton { display: none !important; }

/* ── Header ── */
.app-header {
    padding: 20px 0 12px 0;
    border-bottom: 2px solid #E5E7EB;
    margin-bottom: 24px;
}
.app-header h1 {
    font-size: 1.65rem;
    font-weight: 700;
    color: #111827;
    margin: 0;
}
.app-header p {
    font-size: 0.92rem;
    color: #6B7280;
    margin: 4px 0 0 0;
}

/* ── Step Sections ── */
.step-section {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.step-num {
    background: #2563EB;
    color: #FFFFFF;
    width: 26px; height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
}
.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
}
.step-desc {
    font-size: 0.85rem;
    color: #6B7280;
    margin-bottom: 16px;
}

/* ── Traffic Lights ── */
.traffic-wrap {
    display: flex;
    gap: 18px;
    padding: 16px 20px;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
}
.light-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
}
.light-dot {
    width: 36px; height: 36px;
    border-radius: 50%;
    border: 3px solid #D1D5DB;
    transition: all 0.3s ease;
}
.light-dot.on-green  { background: #22C55E; border-color: #16A34A; box-shadow: 0 0 14px rgba(34,197,94,0.55); }
.light-dot.on-orange { background: #F59E0B; border-color: #D97706; box-shadow: 0 0 14px rgba(245,158,11,0.55); }
.light-dot.on-red    { background: #EF4444; border-color: #DC2626; box-shadow: 0 0 14px rgba(239,68,68,0.55); }
.light-dot.off       { background: #E5E7EB; }
.light-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #6B7280;
}
.light-status {
    font-size: 0.78rem;
    font-weight: 500;
    color: #374151;
}

/* ── Decision Log ── */
.log-entry {
    padding: 9px 14px;
    border-radius: 8px;
    margin-bottom: 5px;
    font-size: 0.88rem;
    line-height: 1.55;
    border-left: 4px solid transparent;
}
.log-entry.safe      { background: #F0FDF4; border-left-color: #22C55E; color: #166534; }
.log-entry.standard  { background: #EFF6FF; border-left-color: #2563EB; color: #1E40AF; }
.log-entry.critical  { background: #FEF2F2; border-left-color: #EF4444; color: #991B1B; }
.log-entry.corrected { background: #FFFBEB; border-left-color: #F59E0B; color: #92400E; }

/* ── Metric Chips ── */
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.chip {
    flex: 1;
    min-width: 120px;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.chip .val {
    font-size: 1.55rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.2;
}
.chip .lbl {
    font-size: 0.7rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 3px;
}

/* ── Progress Bar ── */
.pbar-outer {
    background: #E5E7EB;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin-top: 6px;
}
.pbar-inner {
    height: 100%;
    border-radius: 8px;
    transition: width 0.3s ease;
}
.pbar-inner.lo  { background: #22C55E; }
.pbar-inner.mid { background: #F59E0B; }
.pbar-inner.hi  { background: #EF4444; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PRESETS
# ═══════════════════════════════════════════════════════════════════════

CHANNEL_PRESETS = {
    "☀️  Perfect (No Noise)":              {"ber": 0.0001, "force_bad": False},
    "🌤️  Typical (Low Noise)":             {"ber": 0.008,  "force_bad": False},
    "⛈️  Extreme (Burst Interference)":    {"ber": 0.06,   "force_bad": True},
}
PAYLOAD_PRESETS = {
    "📡 Small (IoT Data)":    64,
    "📄 Medium (Text File)":  256,
    "🖼️ Large (Image Packet)": 512,
}


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def construct_aliasing_error(message_bits, poly_name="CRC-16-CCITT"):
    """Algebraically construct E(x) = Q(x)·G(x)."""
    poly = get_polynomial(poly_name)
    cw = CRCEncoder(poly).encode(message_bits)
    cw_len = len(cw)
    r = poly.degree

    g_bits = []
    pv = poly.poly | (1 << poly.width)
    for i in range(poly.width + 1):
        g_bits.append((pv >> i) & 1)

    qd = r + 2
    q_bits = [0] * (qd + 1)
    q_bits[0] = 1
    q_bits[qd] = 1

    e_len = len(q_bits) + len(g_bits) - 1
    e_bits = [0] * e_len
    for i, qb in enumerate(q_bits):
        if qb:
            for j, gb in enumerate(g_bits):
                if gb:
                    e_bits[i + j] ^= 1

    if len(e_bits) > cw_len:
        e_bits = e_bits[:cw_len]
    else:
        e_bits += [0] * (cw_len - len(e_bits))
    if sum(e_bits) == 0:
        e_bits[r] = 1
        e_bits[r + 1] = 1
    return e_bits


# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

def init_session():
    if "ok" not in st.session_state:
        st.session_state.ok = True
        st.session_state.counter = 0
        st.session_state.log = []
        st.session_state.eff_hist = []
        st.session_state.alias_hist = []
        st.session_state.traffic = "green"
        st.session_state.last_effort = 0.0
        st.session_state.stats = dict(
            total=0, passed=0, corrected_bp=0, corrected_orb=0,
            unrecoverable=0, aliasing=0,
            crc16=0, crc32=0, cksum=0, upgrades=0, energy_saves=0,
        )

        model_path = os.path.join(_PROJECT_ROOT, "outputs", "models", "xgboost_model.joblib")
        xgb = None
        if os.path.exists(model_path):
            import joblib
            xgb = joblib.load(model_path)

        st.session_state.predictor = AliasingPredictor(model=xgb)
        st.session_state.monitor = ChannelStateMonitor(window_size=CHANNEL_WINDOW_SIZE)
        st.session_state.channel = ChannelModel(seed=42)
        st.session_state.rng = random.Random(42)

init_session()


# ═══════════════════════════════════════════════════════════════════════
#  CORE: PROCESS ONE FRAME
# ═══════════════════════════════════════════════════════════════════════

def process_frame(bits, ber, force_bad, inject_alias=False):
    mon = st.session_state.monitor
    pred = st.session_state.predictor
    ch = st.session_state.channel
    S = st.session_state.stats

    st.session_state.counter += 1
    fid = st.session_state.counter
    flen = len(bits)

    if force_bad:
        mon.current_state = "bad"

    noise_type = "burst" if force_bad else "random"
    n_err = max(1, int(ber * flen))
    b_est = max(1, int(ber * flen * 2))

    # ── XGBoost prediction ──
    alias_prob = pred.predict_aliasing_probability(
        frame_length=flen, polynomial_name="CRC-16-CCITT",
        noise_type=noise_type, ber=ber,
        num_errors=n_err, burst_length=b_est,
    )
    if inject_alias:
        alias_prob = max(alias_prob, 0.80)

    # ── Select protection mode ──
    crc16_poly = get_polynomial("CRC-16-CCITT")
    crc32_poly = get_polynomial("CRC-32")
    cksum = InternetChecksum()
    ch_state = mon.current_state

    if ch_state == "good" and alias_prob <= ALIASING_THRESHOLD and ber < 0.005:
        mode, poly, enc, dec = "Checksum", None, None, None
        reason = "Safe Channel Detected: Using Internet Checksum."
        icon, style = "✅", "safe"
        S["cksum"] += 1; S["energy_saves"] += 1
        overhead = 0.4
    elif alias_prob > ALIASING_THRESHOLD:
        mode, poly = "CRC-32", crc32_poly
        enc, dec = CRCEncoder(poly), CRCDecoder(poly)
        reason = f"Critical Aliasing Risk ({alias_prob:.0%}): Switching to Max Protection (CRC-32)."
        icon, style = "🚀", "critical"
        S["crc32"] += 1; S["upgrades"] += 1
        overhead = 1.6
    else:
        mode, poly = "CRC-16", crc16_poly
        enc, dec = CRCEncoder(poly), CRCDecoder(poly)
        reason = "Noise Detected: Standardizing on CRC-16."
        icon, style = "🛡️", "standard"
        S["crc16"] += 1
        overhead = 1.0

    # ── ORBGRAND depth ──
    orb_max = ORBGRAND_AGGRESSIVE if ch_state == "bad" else ORBGRAND_NORMAL

    # ── Encode ──
    codeword = cksum.encode(bits) if mode == "Checksum" else enc.encode(bits)

    # ── Transmit ──
    p_gb  = 0.15 if force_bad else 0.05
    p_bg  = 0.10 if force_bad else 0.50
    p_bad = min(ber * 10, 0.25) if force_bad else ber * 5

    received, llr, err_pat = ch.gilbert_elliott_transmit(
        codeword, p_good=ber, p_bad=p_bad, p_gb=p_gb, p_bg=p_bg,
    )

    # ── Aliasing injection ──
    if inject_alias and mode != "Checksum":
        ae = construct_aliasing_error(bits, poly.name)
        for k in range(min(len(received), len(ae))):
            received[k] ^= ae[k]
        err_pat = ae

    n_errors = sum(err_pat)

    # ── Update monitor ──
    inferred = mon.observe_frame(err_pat, flen, llr)
    if force_bad:
        inferred = "bad"
        mon.current_state = "bad"

    # ── Verify & Correct ──
    result = "PASS"
    reliability = 1.0
    orb_effort = 0.0

    if mode == "Checksum":
        if not cksum.verify(received):
            result = "FAIL"
            reliability = 0.3
        else:
            reliability = 0.9 if n_errors == 0 else 0.5
    else:
        if dec.verify(received):
            if n_errors > 0:
                result = "ALIASING"
                reliability = 0.1
            else:
                result = "PASS"
                reliability = 1.0
        else:
            # ── BP first ──
            bp = BPDecoder(poly, max_iterations=30, damping=0.5)
            bp_dec, bp_ok, bp_it = bp.decode(received, llr)

            if bp_ok:
                result = "CORRECTED_BP"
                reliability = 0.92
                S["corrected_bp"] += 1
            else:
                # ── ORBGRAND with boosted queries & LLR sorting ──
                rel_vals = ch.get_reliability(llr)
                orbgrand = ORBGRANDDecoder(poly, max_queries=orb_max, max_error_weight=5)
                orb_dec, orb_ok, orb_q = orbgrand.decode(received, rel_vals)
                orb_effort = min(orb_q / max(orb_max, 1), 1.0)

                if orb_ok:
                    result = "CORRECTED_ORB"
                    reliability = 0.87
                    S["corrected_orb"] += 1
                else:
                    result = "UNRECOVERABLE"
                    reliability = 0.0
                    S["unrecoverable"] += 1

    if ch_state == "bad":
        overhead *= 1.5

    efficiency = max(0, reliability * (1.0 / max(overhead, 0.1))) * 100
    efficiency = min(efficiency, 100)

    S["total"] += 1
    if result == "PASS":
        S["passed"] += 1
    if result == "ALIASING":
        S["aliasing"] += 1
    st.session_state.last_effort = orb_effort

    # ── Traffic light ──
    if result in ("ALIASING", "UNRECOVERABLE", "FAIL"):
        traffic = "red"
    elif result in ("CORRECTED_BP", "CORRECTED_ORB"):
        traffic = "orange"
        icon, style = "🔧", "corrected"
        method = "BP" if result == "CORRECTED_BP" else "ORBGRAND"
        reason = f"Error Recovered: Frame corrected via {method} ({n_errors} bit errors fixed)."
    else:
        traffic = "green"

    st.session_state.traffic = traffic

    entry = dict(
        fid=fid, time=datetime.now().strftime("%H:%M:%S"),
        icon=icon, reason=reason, style=style, mode=mode,
        result=result, channel=inferred, alias_prob=alias_prob,
        efficiency=efficiency, reliability=reliability,
        orb_queries=orb_max, n_errors=n_errors, traffic=traffic,
    )
    st.session_state.log.append(entry)
    st.session_state.eff_hist.append(efficiency)
    st.session_state.alias_hist.append(alias_prob)

    return entry


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    st.markdown("##### 🕐 Simulation Speed")
    speed = st.slider("Speed", 1, 5, 3, label_visibility="collapsed")
    speed_lbl = {1: "🐌 Very Slow", 2: "🚶 Slow", 3: "🏃 Normal", 4: "🚀 Fast", 5: "⚡ Instant"}
    st.caption(speed_lbl[speed])
    st.markdown("---")

    st.markdown("##### 📡 Channel Condition")
    ch_choice = st.radio("Channel", list(CHANNEL_PRESETS.keys()), index=1, label_visibility="collapsed")
    preset = CHANNEL_PRESETS[ch_choice]
    ber, force_bad = preset["ber"], preset["force_bad"]
    st.markdown("---")

    st.markdown("##### 📦 Payload Type")
    pl_choice = st.selectbox("Payload", list(PAYLOAD_PRESETS.keys()), index=1, label_visibility="collapsed")
    frame_size = PAYLOAD_PRESETS[pl_choice]
    st.markdown("---")

    st.markdown("##### ▶️ Run Simulation")
    c1, c2 = st.columns(2)
    with c1:
        btn1 = st.button("Send 1", key="b1", use_container_width=True)
    with c2:
        btn10 = st.button("Send 10", key="b10", use_container_width=True)
    btn50 = st.button("🔁 Send 50 Frames", key="b50", use_container_width=True)
    st.markdown("---")
    btn_alias = st.button("⚠️ Inject Aliasing Error", key="ba", use_container_width=True)
    st.markdown("---")
    if st.button("🗑️ Reset Session", key="rst", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  PROCESS TRIGGERED FRAMES
# ═══════════════════════════════════════════════════════════════════════

n_send, do_alias = 0, False
if btn1:       n_send = 1
elif btn10:    n_send = 10
elif btn50:    n_send = 50
elif btn_alias: n_send, do_alias = 1, True

for i in range(n_send):
    bits = [st.session_state.rng.randint(0, 1) for _ in range(frame_size)]
    process_frame(bits, ber, force_bad, inject_alias=(do_alias and i == 0))


# ═══════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""<div class="app-header">
    <h1>🛡️ Adaptive Error Control System</h1>
    <p>Real-time intelligent CRC switching · XGBoost aliasing prediction · ORBGRAND self-healing</p>
</div>""", unsafe_allow_html=True)

# ── Summary chips ──
S = st.session_state.stats
total = max(S["total"], 1)
success = S["passed"] + S["corrected_bp"] + S["corrected_orb"]
rel_score = round(success / total * 100)

st.markdown(f"""<div class="chip-row">
    <div class="chip"><div class="val">{S["total"]}</div><div class="lbl">Frames Sent</div></div>
    <div class="chip"><div class="val" style="color:#22C55E">{S["corrected_bp"] + S["corrected_orb"]}</div><div class="lbl">Corrected</div></div>
    <div class="chip"><div class="val" style="color:#DC2626">{S["unrecoverable"]}</div><div class="lbl">Failed</div></div>
    <div class="chip"><div class="val" style="color:#2563EB">{S["upgrades"]}</div><div class="lbl">CRC-32 Upgrades</div></div>
    <div class="chip"><div class="val" style="color:#7C3AED">{S["energy_saves"]}</div><div class="lbl">Energy Saves</div></div>
</div>""", unsafe_allow_html=True)

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 1: INTELLIGENT DECISION HUB
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""<div class="step-section">
    <div class="step-pill"><span class="step-num">1</span><span class="step-title">Intelligent Decision Hub</span></div>
    <p class="step-desc">The AI monitors channel quality and selects the optimal protection level</p>
</div>""", unsafe_allow_html=True)

col_tl, col_log = st.columns([1, 3])

with col_tl:
    # ── Traffic Lights ──
    t = st.session_state.traffic
    g_cls = "on-green"  if t == "green"  else "off"
    o_cls = "on-orange" if t == "orange" else "off"
    r_cls = "on-red"    if t == "red"    else "off"

    g_st = "Active" if t == "green"  else "—"
    o_st = "Active" if t == "orange" else "—"
    r_st = "Active" if t == "red"    else "—"

    st.markdown(f"""
    <div class="traffic-wrap" style="flex-direction:column; align-items:center; padding:20px;">
        <div class="light-item">
            <div class="light-dot {g_cls}"></div>
            <div class="light-label">Verified</div>
            <div class="light-status">{g_st}</div>
        </div>
        <div class="light-item" style="margin-top:10px;">
            <div class="light-dot {o_cls}"></div>
            <div class="light-label">Corrected</div>
            <div class="light-status">{o_st}</div>
        </div>
        <div class="light-item" style="margin-top:10px;">
            <div class="light-dot {r_cls}"></div>
            <div class="light-label">Fail / Aliasing</div>
            <div class="light-status">{r_st}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Active protocol badge ──
    if st.session_state.log:
        last = st.session_state.log[-1]
        mode = last["mode"]
        mc = {"CRC-16": "#2563EB", "CRC-32": "#DC2626", "Checksum": "#16A34A"}.get(mode, "#6B7280")
        st.markdown(f"""
        <div style="text-align:center; margin-top:14px; padding:12px; background:#F9FAFB; border-radius:10px; border:1px solid #E5E7EB;">
            <div style="font-size:0.68rem; color:#6B7280; text-transform:uppercase; letter-spacing:1px;">Protocol</div>
            <div style="font-size:1.2rem; font-weight:700; color:{mc}; margin-top:2px;">{mode}</div>
            <div style="font-size:0.72rem; color:#9CA3AF; margin-top:2px;">ORBGRAND: {last['orb_queries']:,}</div>
        </div>
        """, unsafe_allow_html=True)

with col_log:
    st.markdown("**Decision Log**")
    log = st.session_state.log
    if not log:
        st.info("No frames sent yet. Use the sidebar to begin.")
    else:
        container = st.container(height=310)
        with container:
            for e in reversed(log[-20:]):
                st.markdown(f"""<div class="log-entry {e['style']}">
                    {e['icon']} <strong>#{e['fid']:03d}</strong>
                    <span style="color:#9CA3AF; font-size:0.78rem;">{e['time']}</span>
                    &nbsp;— {e['reason']}
                </div>""", unsafe_allow_html=True)

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 2: SYSTEM RELIABILITY GAUGE + EFFICIENCY CHART
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""<div class="step-section">
    <div class="step-pill"><span class="step-num">2</span><span class="step-title">Transmission & Performance</span></div>
    <p class="step-desc">System Reliability Score tracks successful transmissions; Efficiency Score combines reliability and overhead</p>
</div>""", unsafe_allow_html=True)

col_gauge, col_chart = st.columns([1, 2])

with col_gauge:
    # ── Reliability Gauge (0–100%) ──
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rel_score,
        number={"suffix": "%", "font": {"size": 38, "color": "#111827"}},
        delta={"reference": 80, "increasing": {"color": "#22C55E"}, "decreasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9CA3AF", "tickfont": {"color": "#6B7280"}},
            "bar": {"color": "#2563EB", "thickness": 0.3},
            "bgcolor": "#F3F4F6",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#FEE2E2"},
                {"range": [40, 70], "color": "#FEF3C7"},
                {"range": [70, 100], "color": "#DCFCE7"},
            ],
            "threshold": {"line": {"color": "#111827", "width": 3}, "value": rel_score, "thickness": 0.8},
        },
        title={"text": "System Reliability", "font": {"size": 14, "color": "#374151"}},
    ))
    fig_g.update_layout(
        height=240,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_g, width="stretch", key="gauge")

with col_chart:
    if st.session_state.eff_hist:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            y=st.session_state.eff_hist,
            x=list(range(1, len(st.session_state.eff_hist) + 1)),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.06)",
            line=dict(color="#2563EB", width=2),
            name="Efficiency",
        ))
        fig_e.add_hline(y=50, line_dash="dot", line_color="#D1D5DB",
                        annotation_text="50% baseline", annotation_position="top left",
                        annotation_font_color="#9CA3AF")
        fig_e.update_layout(
            yaxis=dict(title="Efficiency (%)", range=[0, 105], gridcolor="#F3F4F6", zeroline=False),
            xaxis=dict(title="Frame #", gridcolor="#F3F4F6"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=240, margin=dict(l=50, r=20, t=20, b=50),
            font=dict(family="Inter, sans-serif", color="#111827"),
            showlegend=False,
        )
        st.plotly_chart(fig_e, width="stretch", key="eff")
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; background:#F9FAFB; border-radius:10px; border:1px dashed #D1D5DB; color:#9CA3AF;">
            📊 &nbsp; Charts appear after sending frames
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 3: SELF-HEALING DECODER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""<div class="step-section">
    <div class="step-pill"><span class="step-num">3</span><span class="step-title">Self-Healing Decoder (ORBGRAND)</span></div>
    <p class="step-desc">When errors are detected, the decoder guesses the noise pattern using LLR reliability sorting — now boosted to 30K queries</p>
</div>""", unsafe_allow_html=True)

col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    corr_total = S["corrected_bp"] + S["corrected_orb"]
    fail_total = S["unrecoverable"]
    attempted = corr_total + fail_total
    rate = round(corr_total / attempted * 100) if attempted > 0 else 0

    st.markdown(f"""<div style="display:flex; gap:10px;">
        <div class="chip" style="flex:1;"><div class="val" style="color:#22C55E">{corr_total}</div><div class="lbl">Corrected</div></div>
        <div class="chip" style="flex:1;"><div class="val" style="color:#DC2626">{fail_total}</div><div class="lbl">Failed</div></div>
    </div>""", unsafe_allow_html=True)

    rate_color = "#22C55E" if rate >= 60 else ("#F59E0B" if rate >= 30 else "#DC2626")
    st.markdown(f"""
    <div style="margin-top:10px; padding:14px; background:#F9FAFB; border-radius:10px; border:1px solid #E5E7EB; text-align:center;">
        <div style="font-size:0.68rem; color:#6B7280; text-transform:uppercase; letter-spacing:1px;">Recovery Rate</div>
        <div style="font-size:1.5rem; font-weight:700; color:{rate_color}; margin-top:3px;">{rate}%</div>
    </div>""", unsafe_allow_html=True)

with col_s2:
    effort = st.session_state.last_effort
    pct = int(effort * 100)
    bar_cls = "lo" if pct < 30 else ("mid" if pct < 70 else "hi")
    elbl = "Low" if pct < 30 else ("Moderate" if pct < 70 else "High — pushing limits")

    st.markdown(f"""
    <div style="padding:16px 18px; background:#F9FAFB; border-radius:10px; border:1px solid #E5E7EB;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
            <span style="font-weight:600; color:#111827; font-size:0.92rem;">Correction Effort</span>
            <span style="font-size:0.82rem; color:#6B7280;">{elbl}</span>
        </div>
        <div class="pbar-outer">
            <div class="pbar-inner {bar_cls}" style="width:{max(pct,2)}%;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:5px; font-size:0.72rem; color:#9CA3AF;">
            <span>0 queries</span>
            <span>{pct}% budget used</span>
            <span>Max {ORBGRAND_AGGRESSIVE:,}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    recent_corr = [e for e in st.session_state.log[-30:] if e["result"] in ("CORRECTED_BP", "CORRECTED_ORB")]
    if recent_corr:
        st.markdown("")
        st.markdown("**Recent Recoveries**")
        for e in reversed(recent_corr[-5:]):
            method = "BP" if e["result"] == "CORRECTED_BP" else "ORBGRAND"
            st.markdown(f"🔧 Frame **#{e['fid']}** — {e['n_errors']} errors fixed via {method}")
    elif S["total"] > 0:
        st.markdown("")
        st.caption("No corrections needed so far — clean channel.")

st.markdown("")
st.markdown("---")
st.caption("Adaptive Error Control System — Powered by XGBoost, ORBGRAND & Gilbert-Elliott Channel Simulation")
