"""
Generate .docx explaining the full working of the Streamlit deployed model.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "reports", "Streamlit_Model_Working_Explained.docx")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def set_cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_table(doc, headers, rows, header_color="1B3A5C", col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(cell, header_color)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
            if ri % 2 == 1:
                set_cell_shading(cell, "F0F4F8")
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


def code_block(doc, text, sz=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(sz)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F5F5F5"/>')
    p._element.get_or_add_pPr().append(shading)


def note_box(doc, text, label="NOTE"):
    colors = {"NOTE": "E8F4FD", "IMPORTANT": "FFF3E0", "WARNING": "FBE9E7"}
    p = doc.add_paragraph()
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{colors.get(label, "E8F4FD")}"/>')
    p._element.get_or_add_pPr().append(shading)
    r = p.add_run(f"  {label}: ")
    r.bold = True
    r.font.size = Pt(9.5)
    r2 = p.add_run(text)
    r2.font.size = Pt(9.5)


def generate():
    doc = Document()

    for s in doc.sections:
        s.top_margin = Cm(2.5)
        s.bottom_margin = Cm(2.5)
        s.left_margin = Cm(2.5)
        s.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

    for lvl in range(1, 4):
        doc.styles[f"Heading {lvl}"].font.color.rgb = RGBColor(0x0D, 0x27, 0x4D)

    # ═══════════════════════════════════════════════════════════════════
    #  TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════
    for _ in range(6):
        doc.add_paragraph()

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Full Working of the Streamlit\nDeployed Model")
    r.bold = True
    r.font.size = Pt(28)
    r.font.color.rgb = RGBColor(0x0D, 0x27, 0x4D)
    doc.add_paragraph()

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Step-by-Step Execution Trace\nFrom Button Click to Dashboard Update")
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0x34, 0x52, 0x7A)
    doc.add_paragraph()

    lvl = doc.add_paragraph()
    lvl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = lvl.add_run("AI-Driven Adaptive CRC Error Control System\nM.Tech — Computer Networks")
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    #  OVERVIEW
    # ═══════════════════════════════════════════════════════════════════
    doc.add_heading("Overview: What Exactly is Running?", level=1)
    doc.add_paragraph(
        "The Streamlit dashboard is not just a simple visualization tool — it is running a "
        "complete, live simulation of a digital communication system. Every time you click "
        "\"Send Frame,\" the following happens in real-time:"
    )
    code_block(doc, "Your Data → CRC Encode → Noisy Channel → CRC Check → Error Correction → ML Prediction → Dashboard Update")

    doc.add_paragraph("The \"model\" deployed is actually three models working together:")
    add_table(doc,
        ["Component", "Type", "Role"],
        [
            ["XGBoost Classifier", "Machine Learning", "Predicts aliasing probability from channel conditions"],
            ["Belief Propagation (BP)", "Iterative Decoder", "Corrects errors using soft LLR information"],
            ["ORBGRAND", "Universal Decoder", "Noise-guessing decoder (fallback if BP fails)"],
        ],
        col_widths=[4.5, 3.5, 8]
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 0: INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════
    doc.add_heading("Phase 0: Initialization (happens once when page opens)", level=1)

    doc.add_heading("Step 0.1 — Load the Pre-Trained XGBoost Model", level=2)
    doc.add_paragraph(
        "When you first open http://localhost:8501, the function init_session_state() runs. "
        "It loads the XGBoost classifier from outputs/models/xgboost_model.joblib. This model "
        "was trained on 20,000 simulated frames (88% normal + 12% algebraically-engineered "
        "aliasing errors). The model learned which combinations of BER, burst length, error "
        "density, and polynomial degree are most likely to cause CRC aliasing."
    )
    code_block(doc, 'model_path = "outputs/models/xgboost_model.joblib"\nxgb_model = joblib.load(model_path)')

    doc.add_heading("Step 0.2 — Create the Three Controller Components", level=2)
    code_block(doc,
        "monitor   = ChannelStateMonitor(window_size=50)    # Tracks Good/Bad channel state\n"
        "predictor = AliasingPredictor(model=xgb_model)     # Wraps XGBoost for live inference\n"
        "controller = AdaptiveProtocolController(monitor, predictor)  # Decision engine\n"
        "channel   = ChannelModel(seed=42)                  # Gilbert-Elliott channel simulator"
    )

    doc.add_heading("Step 0.3 — Initialize All History Buffers", level=2)
    doc.add_paragraph(
        "Empty lists created for: decision_log, reliability_history, overhead_history, "
        "aliasing_prob_history, state_timeline, crc_mode_timeline, correction_log, "
        "and counters for every statistic (total frames, CRC-32 upgrades, energy saves, etc.)."
    )
    note_box(doc,
        "All of this persists across button clicks via st.session_state — Streamlit "
        "remembers everything until you click 'Reset Session.'",
        "NOTE"
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 1: USER INPUT
    # ═══════════════════════════════════════════════════════════════════
    doc.add_heading("Phase 1: User Input (Sidebar Controls)", level=1)
    doc.add_paragraph(
        "When you interact with the sidebar, you set 5 parameters that control the simulation:"
    )

    doc.add_heading("1.1 — Data Payload", level=2)
    add_table(doc,
        ["Input Mode", "What Happens", "Example"],
        [
            ["Text Message", 'Each character → 8 bits (ASCII)', '"Hello" → 01001000 01100101 ...'],
            ["Custom Bitstream", "Your binary string parsed directly", "10110100 → [1,0,1,1,0,1,0,0]"],
            ["Random (128-bit)", "System generates 128 random 0s and 1s", "[0,1,1,0,1,0,0,1,...]"],
        ],
        col_widths=[4, 6, 6]
    )

    doc.add_heading("1.2 — Initial Protocol Toggle", level=2)
    doc.add_paragraph(
        "OFF (default): Start with CRC-16 — the system may upgrade to CRC-32 if needed. "
        "ON: Start with Internet Checksum — lighter protection, the system will upgrade only if danger is detected."
    )

    doc.add_heading("1.3 — BER (Bit Error Rate) Slider: 0.0001 to 0.10", level=2)
    add_table(doc,
        ["BER Value", "Meaning", "Expected Errors per 128-bit frame"],
        [
            ["0.001", "Clean channel", "~0.13 (rare errors)"],
            ["0.01", "Moderate noise", "~1.3 errors per frame"],
            ["0.05", "Heavy noise", "~6.4 errors per frame"],
            ["0.10", "Extremely noisy", "~12.8 errors per frame"],
        ]
    )

    doc.add_heading("1.4 — Force Gilbert-Elliott 'Bad' State", level=2)
    doc.add_paragraph(
        "OFF: Channel naturally transitions between Good ↔ Bad states. "
        "ON: Forces the channel to stay in the Bad state permanently (higher error rate, bursty errors, "
        "10× the BER, transitions biased towards staying Bad)."
    )

    doc.add_heading("1.5 — 'INJECT CRITICAL FAIL' Button", level=2)
    doc.add_paragraph(
        "This triggers algebraic aliasing engineering — constructs a special error pattern "
        "E(x) = Q(x) · G(x) that is mathematically guaranteed to bypass CRC detection. "
        "Since E(x) is a multiple of the generator polynomial G(x), dividing by G(x) yields "
        "remainder = 0. The CRC check passes even though the data is corrupted."
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 2: FRAME PROCESSING (the core)
    # ═══════════════════════════════════════════════════════════════════
    doc.add_heading("Phase 2: Frame Processing (the core engine)", level=1)
    doc.add_paragraph(
        'When you click "Send 1 Frame" (or 25/100), the function process_single_frame() '
        "executes. Here's exactly what happens, step by step:"
    )

    # Step 2.1
    doc.add_heading("Step 2.1 — Estimate Channel Conditions", level=2)
    doc.add_paragraph(
        "The system estimates how many errors the channel will likely introduce based on the BER you set:"
    )
    code_block(doc,
        'noise_type = "burst" if force_bad_state else "random"\n'
        "num_errors_est = max(1, int(ber * frame_len))        # e.g., 0.01 × 128 = 1\n"
        "burst_length_est = max(1, int(ber * frame_len * 2))  # e.g., 0.01 × 128 × 2 = 2"
    )

    # Step 2.2
    doc.add_heading("Step 2.2 — XGBoost Aliasing Prediction (THE ML MODEL)", level=2)
    doc.add_paragraph(
        "This is where the trained XGBoost model runs inference. The predictor:"
    )

    doc.add_heading("Step A — Build a 14-dimensional feature vector:", level=3)
    add_table(doc,
        ["#", "Feature Name", "Calculation", "Example Value"],
        [
            ["1", "frame_length", "Length of user's data", "128"],
            ["2", "codeword_length", "frame_length + polynomial_degree", "144"],
            ["3", "polynomial_degree", "Degree of G(x)", "16"],
            ["4", "polynomial_type_encoded", "CCITT=0, IBM=1, CRC-32=2", "0"],
            ["5", "noise_type_encoded", "burst=0, multi=1, random=2, single=3", "2"],
            ["6", "ber", "User's BER slider value", "0.01"],
            ["7", "burst_length_actual", "Estimated burst length", "2"],
            ["8", "num_errors_injected", "Estimated error count", "1"],
            ["9", "error_density", "errors / codeword_length", "0.0069"],
            ["10", "error_to_poly_ratio", "burst_length / degree", "0.125"],
            ["11", "burst_exceeds_degree", "1 if burst > degree, else 0", "0"],
            ["12", "error_weight_ratio", "min(errors / degree, 10)", "0.0625"],
            ["13", "frame_poly_ratio", "frame_length / degree", "8.0"],
            ["14", "ber_x_frame_length", "BER × frame_length", "1.28"],
        ],
        col_widths=[1, 4.5, 5.5, 3]
    )

    doc.add_heading("Step B — Feed to XGBoost's predict_proba():", level=3)
    code_block(doc,
        "proba = self.model.predict_proba(features)   # Returns [[P(undetected), P(detected)]]\n"
        "aliasing_prob = float(proba[0, 0])           # e.g., 0.004 (0.4%)"
    )

    doc.add_paragraph(
        "The model outputs two probabilities: proba[0,0] = probability the error will be "
        "undetected (aliasing), and proba[0,1] = probability the error will be detected. "
        "We use the aliasing probability to make the switching decision."
    )

    doc.add_heading("Example XGBoost Predictions:", level=3)
    add_table(doc,
        ["BER", "Burst Length", "Error Density", "XGBoost Output", "Action"],
        [
            ["0.001", "2", "0.007", "0.4% (safe)", "Stay on CRC-16"],
            ["0.05", "20", "0.08", "22% (dangerous)", "UPGRADE to CRC-32!"],
            ["0.10", "40", "0.15", "99.5% (critical)", "CRC-32 + ORBGRAND 15K"],
        ],
        header_color="8B0000"
    )

    # Step 2.3
    doc.add_heading("Step 2.3 — Adaptive Switching Decision", level=2)
    doc.add_paragraph(
        "Based on the XGBoost prediction, the controller decides which protection to use:"
    )

    add_table(doc,
        ["Condition", "Action Taken", "Reason Logged"],
        [
            ["aliasing_prob > 15%", "Mode = CRC-32", "⚠️ Aliasing Prob X% > 15% → UPGRADING to CRC-32"],
            ["Checksum mode + good + safe", "Mode = Internet Checksum", "⚡ Energy-Efficiency: overhead ↓ 40%"],
            ["Otherwise (normal)", "Mode = CRC-16", "✅ Standard CRC-16 (aliasing prob = X%)"],
            ["channel_state == 'bad'", "ORBGRAND: 5K → 15K", "🔄 Bad channel → ORBGRAND ↑ 15,000"],
        ]
    )

    # Step 2.4
    doc.add_heading("Step 2.4 — CRC Encoding", level=2)
    doc.add_paragraph("The message bits are encoded with the chosen protection:")

    add_table(doc,
        ["Mode", "Polynomial", "Message → Codeword", "Overhead"],
        [
            ["CRC-32", "G(x) = 0x04C11DB7, degree 32", "128 bits → 160 bits", "1.5× (high)"],
            ["CRC-16", "G(x) = 0x1021, degree 16", "128 bits → 144 bits", "1.0× (standard)"],
            ["Checksum", "RFC 1071 one's complement", "128 bits → 144 bits", "0.6× (low)"],
        ]
    )

    doc.add_paragraph("Inside CRC encode():")
    for step in [
        "1. Multiply M(x) by x^r — append r zero bits to the message",
        "2. Modulo-2 long division — divide by G(x) using XOR operations",
        "3. The remainder R(x) = the CRC checksum",
        "4. Output: T(x) = message_bits + R(x) — ensures T(x) mod G(x) = 0",
    ]:
        doc.add_paragraph(step, style="List Bullet")

    # Step 2.5
    doc.add_heading("Step 2.5 — Channel Transmission (Gilbert-Elliott)", level=2)
    doc.add_paragraph(
        "The encoded codeword is transmitted through a simulated noisy channel:"
    )
    code_block(doc,
        "received, llr, error_pattern = channel.gilbert_elliott_transmit(\n"
        "    codeword,\n"
        "    p_good = ber,         # Error rate in Good state\n"
        "    p_bad  = ber * 10,    # Error rate in Bad state (10× worse)\n"
        "    p_gb   = 0.05,        # Probability of Good → Bad transition\n"
        "    p_bg   = 0.5,         # Probability of Bad → Good transition\n"
        ")"
    )

    doc.add_paragraph("What happens inside the channel, for each bit position i:")
    for step in [
        "1. Check current state (Good or Bad)",
        "2. Flip the bit with probability p_good or p_bad",
        "3. Compute LLR: log((1-p)/p) if received=0, else -log((1-p)/p)",
        "4. Randomly transition state: Good→Bad with p_gb, Bad→Good with p_bg",
    ]:
        doc.add_paragraph(step, style="List Bullet")

    doc.add_paragraph("Output:")
    add_table(doc,
        ["Output", "Type", "Description"],
        [
            ["received", "List[int]", "The corrupted codeword (some bits flipped)"],
            ["llr", "List[float]", "Soft reliability per bit (how confident we are)"],
            ["error_pattern", "List[int]", "Which bits were actually flipped (for analysis)"],
        ]
    )

    # Step 2.6
    doc.add_heading("Step 2.6 — Aliasing Injection (only if CRITICAL FAIL pressed)", level=2)
    doc.add_paragraph(
        "If you pressed the red button, the system constructs a mathematically undetectable error:"
    )
    code_block(doc,
        "# Construct E(x) = Q(x) · G(x)\n"
        "# 1. Take the polynomial G(x) = 0x1021 as bit list\n"
        "# 2. Choose random quotient Q(x)\n"
        "# 3. Multiply: E(x) = Q(x) × G(x) in GF(2) — XOR-based multiplication\n"
        "# 4. XOR the error onto the received codeword\n"
        "for k in range(len(received)):\n"
        "    received[k] ^= alias_error[k]"
    )
    doc.add_paragraph(
        "Since E(x) is a multiple of G(x), the CRC check computes: "
        "T'(x) mod G(x) = (T(x) ⊕ E(x)) mod G(x) = 0 ⊕ 0 = 0. "
        "The error is invisible to CRC."
    )

    # Step 2.7
    doc.add_heading("Step 2.7 — Channel State Monitoring", level=2)
    doc.add_paragraph(
        "The error pattern is fed to the ChannelStateMonitor's sliding window (last 50 frames). "
        "It counts errors, computes moving-average BER, and if BER > 5% or LLR values indicate "
        "low reliability, the state is classified as 'bad'. Otherwise 'good'."
    )

    # Step 2.8
    doc.add_heading("Step 2.8 — CRC Verification & Error Correction (THE DECODERS)", level=2)
    doc.add_paragraph("This is the detection and correction pipeline:")

    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Detection & Correction Cascade:")
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0x0D, 0x27, 0x4D)

    code_block(doc,
        "                    received codeword\n"
        "                          │\n"
        "                    ┌─────▼─────┐\n"
        "                    │ CRC Check │  (divide by G(x), check remainder)\n"
        "                    └─────┬─────┘\n"
        "                          │\n"
        "              ┌───────────┼───────────┐\n"
        "              │           │           │\n"
        "         Remainder=0  Remainder=0  Remainder≠0\n"
        "         & no errors  & errors      │\n"
        "              │         exist        │\n"
        "              ▼           ▼          ▼\n"
        "           ✅ PASS    💀 ALIASING   Error Detected!\n"
        "                    (undetected     │\n"
        "                     error!)        │\n"
        '                              ┌─────▼─────┐\n'
        '                              │ BP Decoder │  (30 iterations)\n'
        '                              └─────┬─────┘\n'
        '                                    │\n'
        '                          ┌─────────┼─────────┐\n'
        '                          │                   │\n'
        '                     BP Success          BP Failed\n'
        '                          │                   │\n'
        '                          ▼                   ▼\n'
        '                    🔵 CORRECTED_BP    ┌─────────────┐\n'
        '                                       │  ORBGRAND   │\n'
        '                                       └──────┬──────┘\n'
        '                                              │\n'
        '                                    ┌─────────┼─────────┐\n'
        '                                    │                   │\n'
        '                               ORBGRAND OK        ORBGRAND Fail\n'
        '                                    │                   │\n'
        '                                    ▼                   ▼\n'
        '                              🟠 CORRECTED      ❌ UNRECOVERABLE',
        sz=7,
    )

    doc.add_paragraph()
    doc.add_heading("BP Decoder — How it works:", level=3)
    for step in [
        "1. Builds a parity-check matrix H from the CRC polynomial (r × n binary matrix)",
        "2. Runs iterative message-passing on the Tanner graph (sum-product algorithm)",
        "3. Check-to-variable update: each check node computes 2·atanh(∏ tanh(msg/2))",
        "4. Variable-to-check update: each variable sums channel LLR + all incoming check messages",
        "5. After each iteration, performs hard decision and checks CRC — if pass, terminates early",
        "6. Maximum 30 iterations with 0.5 damping factor for convergence on dense CRC matrices",
    ]:
        doc.add_paragraph(step, style="List Bullet")

    doc.add_heading("ORBGRAND Decoder — How it works (if BP fails):", level=3)
    for step in [
        "1. Sorts all bit positions by reliability: least reliable first (using |LLR| values)",
        "2. Weight 1: tries flipping each of the least reliable bits, one at a time — checks CRC",
        "3. Weight 2: tries flipping pairs of least reliable bits — checks CRC after each flip",
        "4. Weight 3, 4, 5: continues with increasing combinations",
        "5. For each candidate: if CRC passes → declares success and outputs corrected codeword",
        "6. Budget: 5,000 queries (normal/good channel) or 15,000 queries (bad channel — more attempts)",
    ]:
        doc.add_paragraph(step, style="List Bullet")

    add_table(doc,
        ["Error Weight", "Correction Rate", "Avg Queries Used"],
        [
            ["1 bit", "~95%", "~5"],
            ["2 bits", "~80%", "~50"],
            ["3 bits", "~60%", "~500"],
            ["4 bits", "~35%", "~3,000"],
            ["5 bits", "~15%", "~8,000"],
        ]
    )

    # Step 2.9
    doc.add_heading("Step 2.9 — Traffic Light Assignment", level=2)
    add_table(doc,
        ["Condition", "Traffic Light", "Meaning"],
        [
            ["Aliasing injection or undetected error", "🔴 RED", "Data integrity compromised"],
            ["Corrected by BP or ORBGRAND", "🟠 ORANGE", "Error detected and recovered"],
            ["Unrecoverable error", "🔴 RED", "Error too severe to correct"],
            ["Clean pass (no errors)", "🟢 GREEN", "Data integrity verified"],
        ]
    )

    # Step 2.10
    doc.add_heading("Step 2.10 — Store Everything for Dashboard", level=2)
    doc.add_paragraph(
        "All results are appended to session state history lists:"
    )
    code_block(doc,
        "decision_log.append({frame_id, timestamp, channel_state, alias_prob, crc_mode, ...})\n"
        "reliability_history.append(reliability)      # For the Reliability line chart\n"
        "overhead_history.append(overhead_base)        # For the Overhead line chart\n"
        "aliasing_prob_history.append(alias_prob)      # For the Threshold timeline\n"
        "crc_mode_timeline.append(crc_mode)            # For protocol tracking"
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    #  PHASE 3: DASHBOARD RENDERING
    # ═══════════════════════════════════════════════════════════════════
    doc.add_heading("Phase 3: Dashboard Rendering", level=1)
    doc.add_paragraph(
        "After process_single_frame() returns, Streamlit re-renders the entire page with updated data:"
    )

    doc.add_heading("3.1 — Status Cards (top row)", level=2)
    add_table(doc,
        ["Card", "Shows", "Source"],
        [
            ["Active Polynomial", "CRC-16 / CRC-32 / Checksum badge", 'latest["crc_mode"]'],
            ["ORBGRAND Depth", "5,000 or 15,000", 'latest["orb_queries"]'],
            ["Channel State", "GOOD (green) or BAD (red)", 'latest["channel_state"]'],
            ["System Status", "Traffic light (3 glowing circles)", "current_traffic_light"],
        ]
    )

    doc.add_heading("3.2 — Statistics Row", level=2)
    add_table(doc,
        ["Metric", "Calculation"],
        [
            ["Total Frames", 'stats["total"]'],
            ["CRC-32 Upgrades", "Count and percentage of upgrade decisions"],
            ["Energy Saves", "Count and percentage of Checksum usage"],
            ["Recoveries", "BP corrections + ORBGRAND corrections"],
            ["Avg Reliability", "mean(reliability_history)"],
        ]
    )

    doc.add_heading("3.3 — Decision Log", level=2)
    doc.add_paragraph(
        "A scrollable container showing the last 40 entries, each with emoji icon, "
        "frame number, timestamp, result status, and the controller's reasoning."
    )
    code_block(doc,
        '🟢 #001 02:05:04 [PASS]          — ✅ Standard CRC-16 (aliasing prob = 0.41%)\n'
        '🟠 #008 02:05:06 [CORRECTED_ORB]  — ✅ Standard CRC-16 | 🔄 ORBGRAND ↑ 5,000\n'
        '❌ #015 02:05:08 [UNRECOVERABLE]  — ⚠️ Aliasing Prob 99.5% → UPGRADING to CRC-32'
    )

    doc.add_heading("3.4 — Plotly Charts", level=2)

    doc.add_heading("Chart 1: Reliability vs. Computational Overhead", level=3)
    doc.add_paragraph(
        "Dual Y-axis Plotly line chart. Left axis: Reliability score (0 to 1) as teal line "
        "with area fill. Right axis: Overhead cost as orange dashed line. X-axis: Frame number."
    )

    doc.add_heading("Chart 2: XGBoost Aliasing Probability Gauge", level=3)
    doc.add_paragraph(
        "Semicircular Plotly gauge from 0% to 100%. Green zone (0-15%), Yellow zone (15-50%), "
        "Red zone (50-100%). Shows current aliasing probability and delta from previous frame."
    )

    doc.add_heading("Chart 3: Aliasing Probability Over Time", level=3)
    doc.add_paragraph(
        "Time series: blue line = XGBoost's prediction per frame; orange dashed = 15% threshold; "
        "red shaded regions = frames where system upgraded to CRC-32."
    )

    doc.add_heading("3.5 — Correction Log Table", level=2)
    doc.add_paragraph(
        "Shows frames where BP or ORBGRAND successfully corrected errors. Displays frame number, "
        "method used, error count, first 32 bits of original, first 32 bits of recovered, "
        "and whether they match."
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    #  WHAT MAKES IT INTELLIGENT
    # ═══════════════════════════════════════════════════════════════════
    doc.add_heading("What Makes This 'Intelligent'?", level=1)
    doc.add_paragraph(
        "The key difference from a textbook CRC system is the feedback loop:"
    )

    add_table(doc,
        ["Aspect", "Traditional CRC", "Our Deployed Model"],
        [
            ["Polynomial", "Fixed (always CRC-16)", "Dynamically switches CRC-16 / CRC-32 / Checksum"],
            ["Prediction", "No prediction", "XGBoost predicts danger BEFORE it happens"],
            ["Error handling", "Detect only, discard on error", "BP + ORBGRAND attempt live correction"],
            ["Parameters", "Fixed query depth", "ORBGRAND adapts: 5K ↔ 15K queries"],
            ["Channel awareness", "None", "Gilbert-Elliott monitor tracks channel health"],
            ["User visibility", "None", "Live dashboard with every decision + reasoning"],
        ]
    )

    doc.add_paragraph()
    note_box(doc,
        "The XGBoost model is the decision trigger — it runs on every single frame, "
        "evaluates 14 features in microseconds, and tells the controller: 'There's a 22% "
        "chance this frame's error will slip through CRC-16 undetected. Upgrade now.'",
        "IMPORTANT"
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    doc.add_heading("Summary: One Frame in 10 Steps", level=1)

    add_table(doc,
        ["Step", "What Happens", "Key Component"],
        [
            ["1", "User sets BER, channel state, payload", "Sidebar widgets"],
            ["2", "Estimate noise conditions (type, errors, burst)", "process_single_frame()"],
            ["3", "XGBoost predicts aliasing probability", "AliasingPredictor.predict_aliasing_probability()"],
            ["4", "Controller decides: CRC-16 / CRC-32 / Checksum", "Switching logic (if/elif/else)"],
            ["5", "Controller decides: ORBGRAND 5K or 15K", "Channel state check"],
            ["6", "CRC encode the message", "CRCEncoder.encode()"],
            ["7", "Transmit through Gilbert-Elliott channel", "ChannelModel.gilbert_elliott_transmit()"],
            ["8", "CRC verify → if fail → BP → ORBGRAND", "BPDecoder + ORBGRANDDecoder"],
            ["9", "Set traffic light 🟢 🟠 🔴", "Traffic light assignment"],
            ["10", "Log everything, update charts", "Session state + Plotly re-render"],
        ],
        col_widths=[1.5, 6, 8.5]
    )

    doc.add_paragraph()
    f = doc.add_paragraph()
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = f.add_run("M.Tech Computer Networks — Streamlit Model Working Explained")
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    # ── Save ──
    doc.save(OUTPUT_PATH)
    print(f"\n{'='*60}")
    print(f"  Report generated successfully!")
    print(f"  Path: {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate()
