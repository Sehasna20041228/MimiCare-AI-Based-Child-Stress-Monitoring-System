"""
app_streamlit.py — Mimi AI Caregiver (Streamlit version)
Deploy : Streamlit Community Cloud
Run    : streamlit run app_streamlit.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from cv_core import analyse_photo, analyse_video

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mimi AI Caregiver",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DISCLAIMER ───────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ Mimi is a caregiver support tool only — not a clinical or diagnostic tool. "
    "Always consult a qualified healthcare professional or autism specialist."
)

# ── MIMI SVG CHARACTERS ──────────────────────────────────────────────────────
MIMI_CALM_SVG = """
<div style="text-align:center;padding:10px 0;">
<svg width="100" height="100" viewBox="0 0 90 90" xmlns="http://www.w3.org/2000/svg">
  <style>
    @keyframes bob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }
    .mimi-body { animation: bob 2s ease-in-out infinite; }
  </style>
  <g class="mimi-body">
    <circle cx="45" cy="45" r="40" fill="#FFF3E0" stroke="#FFB74D" stroke-width="2"/>
    <ellipse cx="45" cy="52" rx="18" ry="14" fill="#FFCC80"/>
    <circle cx="35" cy="40" r="5" fill="white" stroke="#90CAF9" stroke-width="1.5"/>
    <circle cx="55" cy="40" r="5" fill="white" stroke="#90CAF9" stroke-width="1.5"/>
    <circle cx="36" cy="40" r="2.5" fill="#1565C0"/>
    <circle cx="56" cy="40" r="2.5" fill="#1565C0"/>
    <path d="M36 52 Q45 60 54 52" stroke="#E65100" stroke-width="2" fill="none" stroke-linecap="round"/>
    <ellipse cx="30" cy="44" rx="5" ry="3" fill="#FFAB91" opacity="0.6"/>
    <ellipse cx="60" cy="44" rx="5" ry="3" fill="#FFAB91" opacity="0.6"/>
    <path d="M20 28 Q25 18 35 22" stroke="#FFB74D" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M70 28 Q65 18 55 22" stroke="#FFB74D" stroke-width="3" fill="none" stroke-linecap="round"/>
  </g>
</svg>
<div style="font-size:14px;font-weight:600;color:#E65100;margin-top:4px;">Mimi is here 🌟</div>
</div>
"""

MIMI_HAPPY_SVG = """
<div style="text-align:center;padding:10px 0;">
<svg width="100" height="100" viewBox="0 0 90 90" xmlns="http://www.w3.org/2000/svg">
  <style>
    @keyframes jump { 0%,100%{transform:translateY(0)} 40%{transform:translateY(-5px)} 60%{transform:translateY(-3px)} }
    .mimi-body { animation: jump 1.2s ease-in-out infinite; }
  </style>
  <g class="mimi-body">
    <circle cx="45" cy="45" r="40" fill="#E8F5E9" stroke="#66BB6A" stroke-width="2"/>
    <ellipse cx="45" cy="52" rx="18" ry="14" fill="#A5D6A7"/>
    <circle cx="35" cy="40" r="5" fill="white" stroke="#81C784" stroke-width="1.5"/>
    <circle cx="55" cy="40" r="5" fill="white" stroke="#81C784" stroke-width="1.5"/>
    <circle cx="36" cy="40" r="2.5" fill="#1B5E20"/>
    <circle cx="56" cy="40" r="2.5" fill="#1B5E20"/>
    <path d="M34 52 Q45 63 56 52" stroke="#2E7D32" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <ellipse cx="30" cy="45" rx="5" ry="3" fill="#EF9A9A" opacity="0.7"/>
    <ellipse cx="60" cy="45" rx="5" ry="3" fill="#EF9A9A" opacity="0.7"/>
    <path d="M22 30 Q18 22 26 20" stroke="#66BB6A" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <path d="M68 30 Q72 22 64 20" stroke="#66BB6A" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  </g>
</svg>
<div style="font-size:14px;font-weight:600;color:#2E7D32;margin-top:4px;">Mimi is happy! ✅</div>
</div>
"""

MIMI_ALERT_SVG = """
<div style="text-align:center;padding:10px 0;">
<svg width="100" height="100" viewBox="0 0 90 90" xmlns="http://www.w3.org/2000/svg">
  <style>
    @keyframes shake { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-3px)} 75%{transform:translateX(3px)} }
    .mimi-body { animation: shake 0.4s ease-in-out infinite; }
  </style>
  <g class="mimi-body">
    <circle cx="45" cy="45" r="40" fill="#FFF8E1" stroke="#FF8F00" stroke-width="2.5"/>
    <ellipse cx="45" cy="52" rx="18" ry="14" fill="#FFD54F"/>
    <circle cx="35" cy="40" r="5" fill="white" stroke="#EF9A9A" stroke-width="1.5"/>
    <circle cx="55" cy="40" r="5" fill="white" stroke="#EF9A9A" stroke-width="1.5"/>
    <circle cx="35" cy="39" r="3" fill="#B71C1C"/>
    <circle cx="55" cy="39" r="3" fill="#B71C1C"/>
    <path d="M38 54 Q45 50 52 54" stroke="#BF360C" stroke-width="2" fill="none" stroke-linecap="round"/>
    <path d="M30 33 L34 29 M38 31 L36 27" stroke="#FF8F00" stroke-width="2" stroke-linecap="round"/>
    <path d="M60 33 L56 29 M52 31 L54 27" stroke="#FF8F00" stroke-width="2" stroke-linecap="round"/>
  </g>
</svg>
<div style="font-size:14px;font-weight:600;color:#BF360C;margin-top:4px;">Mimi is worried 🚨</div>
</div>
"""


# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1rem; max-width: 1100px; }
  .mimi-header {
    background: linear-gradient(135deg, #FFF3E0, #E8F5E9);
    border-radius: 16px; padding: 18px 24px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 16px;
  }
  .result-green { background:#E8F5E9; border-left:5px solid #43A047; padding:12px 16px; border-radius:8px; margin:8px 0; }
  .result-yellow { background:#FFFDE7; border-left:5px solid #F9A825; padding:12px 16px; border-radius:8px; margin:8px 0; }
  .result-red { background:#FFEBEE; border-left:5px solid #E53935; padding:12px 16px; border-radius:8px; margin:8px 0; }
  .chat-msg-user { background:#E3F2FD; border-radius:12px 12px 4px 12px; padding:8px 14px; margin:4px 0; text-align:right; }
  .chat-msg-mimi { background:#F3E5F5; border-radius:12px 12px 12px 4px; padding:8px 14px; margin:4px 0; }
  div[data-testid="stRadio"] label { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mimi-header">
  <svg width="70" height="70" viewBox="0 0 90 90" xmlns="http://www.w3.org/2000/svg">
    <circle cx="45" cy="45" r="40" fill="#FFF3E0" stroke="#FFB74D" stroke-width="2"/>
    <ellipse cx="45" cy="52" rx="18" ry="14" fill="#FFCC80"/>
    <circle cx="35" cy="40" r="5" fill="white" stroke="#90CAF9" stroke-width="1.5"/>
    <circle cx="55" cy="40" r="5" fill="white" stroke="#90CAF9" stroke-width="1.5"/>
    <circle cx="36" cy="40" r="2.5" fill="#1565C0"/>
    <circle cx="56" cy="40" r="2.5" fill="#1565C0"/>
    <path d="M36 52 Q45 60 54 52" stroke="#E65100" stroke-width="2" fill="none" stroke-linecap="round"/>
    <ellipse cx="30" cy="44" rx="5" ry="3" fill="#FFAB91" opacity="0.6"/>
    <ellipse cx="60" cy="44" rx="5" ry="3" fill="#FFAB91" opacity="0.6"/>
    <animateTransform attributeName="transform" type="translate" values="0,0;0,-2;0,0" dur="2s" repeatCount="indefinite"/>
  </svg>
  <div>
    <div style="font-size:26px;font-weight:700;color:#E65100;">🌟 Mimi AI Caregiver</div>
    <div style="font-size:14px;color:#5D4037;">Supporting autistic children's emotional wellbeing</div>
    <div style="font-size:12px;color:#8D6E63;margin-top:4px;">⚠️ Caregiver support tool only — not a clinical or diagnostic tool. Always consult a healthcare professional.</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── SCORING HELPERS ──────────────────────────────────────────────────────────
def score_checklist(sleep, comm, stim, eating, sensory, routine, meltdown, new_beh):
    s = 0
    s += {"Much worse than usual": 3, "Slightly worse": 2, "About the same": 0, "Better than usual": 0}.get(sleep, 0)
    s += {"Not communicating at all": 3, "Much less than usual": 2, "Slightly reduced": 1, "About the same": 0}.get(comm, 0)
    s += {"About the same": 0, "Slightly more": 1, "Significantly more": 2, "Extremely intense / distressing": 3}.get(stim, 0)
    s += {"Eating normally": 0, "Slightly reduced": 1, "Refusing some foods": 2, "Refusing to eat": 3}.get(eating, 0)
    s += {"No more than usual": 0, "Slightly more sensitive": 1, "Noticeably more sensitive": 2, "Covering ears / avoiding touch": 3}.get(sensory, 0)
    s += {"No disruption": 0, "Minor change": 1, "Moderate disruption": 2, "Major disruption": 3}.get(routine, 0)
    s += {"No signs at all": 0, "Mild signs — quieter or more rigid": 1, "Clear signs — crying, refusing, intense rocking": 3, "Already in meltdown or shutdown": 4}.get(meltdown, 0)
    s += {"No": 0, "Minor — slightly different": 1, "Yes — not seen before": 2}.get(new_beh, 0)
    return s


def result_label(pred):
    if pred == 0:
        return ("🟢 Well Regulated", "Child appears calm and settled.",
                "✅ Maintain today's routine\n✅ Continue familiar activities\n✅ Keep environment calm\n⏰ Check again in 4–6 hours",
                "result-green")
    elif pred == 1:
        return ("🟡 Mild to Moderate Stress", "Some dysregulation observed.",
                "💛 Offer a calm, low-stimulation space\n💛 Stick to familiar routines\n💛 Use preferred calming strategies\n💛 Speak in short, clear sentences\n⏰ Check again within 1–2 hours",
                "result-yellow")
    else:
        return ("🔴 High Stress / Distress", "Immediate support needed.",
                "🚨 Stay close and stay calm yourself\n🚨 Reduce all sensory input immediately\n🚨 Avoid demands during meltdown/shutdown\n🚨 Offer comfort items silently\n🚨 Contact GP if distress continues",
                "result-red")


# ── CHATBOT ──────────────────────────────────────────────────────────────────
CHAT_RULES = [
    (["meltdown", "shutdown", "crisis"],
     "During a meltdown, reduce demands and sensory input. Stay calm, speak as little as possible, offer a safe space. Avoid restraint unless immediate danger. Allow full recovery time. 🛡️"),
    (["stim", "stimming", "rocking", "flapping"],
     "Stimming is natural self-regulation — never eliminate it entirely. Only redirect if it causes harm, and offer a safe alternative like a fidget toy. 🌀"),
    (["sensory", "noise", "light", "texture", "overwhelm"],
     "Reduce stimulation immediately: dim lights, noise-cancelling headphones, quiet room. A sensory diet of regular planned breaks prevents overload. 🎧"),
    (["routine", "transition", "change", "schedule"],
     "Use visual schedules and give advance warnings before transitions. Stay calm and validate their reaction. 📅"),
    (["communicate", "speech", "nonverbal", "aac"],
     "Accept all forms of communication — pointing, gestures, AAC devices. Never pressure verbal responses during stress. 🗣️"),
    (["sleep"],
     "A consistent bedtime routine helps most. Reduce screens 1 hour before bed. Weighted blankets and white noise work well for many autistic children. 😴"),
    (["eat", "food", "meal", "appetite"],
     "Food selectivity is common — offer familiar safe foods without pressure. Never force eating. A dietitian experienced in autism can help. 🍽️"),
    (["calm", "calming", "regulate", "settle"],
     "Try: weighted blankets, quiet safe space, preferred sensory items, or slow breathing modelled by you. 🌿"),
    (["anxious", "anxiety", "fear", "scared"],
     "Validate feelings first — 'I can see this feels scary.' Predictability and advance preparation reduce anxiety. 💙"),
    (["help"],
     "I can help with: meltdowns, stimming, sensory overload, routines, communication, sleep, eating, and calming strategies. Just ask! 🌟"),
]


def get_reply(msg):
    m = msg.lower()
    for keywords, response in CHAT_RULES:
        if any(w in m for w in keywords):
            return response
    return "Try asking about meltdowns, stimming, sensory needs, routines, sleep, or calming strategies. 😊"


def render_chatbot(key):
    """Render a lightweight chatbot using session_state."""
    hist_key = f"chat_hist_{key}"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    # Display chat history
    for user_msg, mimi_msg in st.session_state[hist_key]:
        st.markdown(f'<div class="chat-msg-user">👤 {user_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-msg-mimi">🌟 {mimi_msg}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        user_input = st.text_input("Ask Mimi:", key=f"input_{key}", placeholder="e.g. How do I help during a meltdown?", label_visibility="collapsed")
    with col2:
        if st.button("Send 💬", key=f"send_{key}"):
            if user_input.strip():
                reply = get_reply(user_input)
                st.session_state[hist_key].append((user_input, reply))
                st.rerun()
    with col3:
        if st.button("Clear", key=f"clear_{key}"):
            st.session_state[hist_key] = []
            st.rerun()


# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Behaviour Checklist",
    "📷 Photo Analysis (CV)",
    "🎥 Video Analysis (CV)",
    "📋+📷+🎥 Combined",
])

SLEEP_OPTS    = ["About the same", "Better than usual", "Slightly worse", "Much worse than usual"]
COMM_OPTS     = ["About the same", "Slightly reduced", "Much less than usual", "Not communicating at all"]
STIM_OPTS     = ["About the same", "Slightly more", "Significantly more", "Extremely intense / distressing"]
EAT_OPTS      = ["Eating normally", "Slightly reduced", "Refusing some foods", "Refusing to eat"]
SENSORY_OPTS  = ["No more than usual", "Slightly more sensitive", "Noticeably more sensitive", "Covering ears / avoiding touch"]
ROUTINE_OPTS  = ["No disruption", "Minor change", "Moderate disruption", "Major disruption"]
MELTDOWN_OPTS = ["No signs at all", "Mild signs — quieter or more rigid", "Clear signs — crying, refusing, intense rocking", "Already in meltdown or shutdown"]
NEWBEH_OPTS   = ["No", "Minor — slightly different", "Yes — not seen before"]


# ── TAB 1: CHECKLIST ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("**Answer based on today compared to your child's usual baseline.**")
    col_a, col_b = st.columns(2)
    with col_a:
        sleep    = st.radio("🌙 Sleep quality last night",   SLEEP_OPTS,    index=0, key="t1_sleep")
        comm     = st.radio("💬 Communication level today",  COMM_OPTS,     index=0, key="t1_comm")
        stim     = st.radio("🔄 Stimming / repetitive",      STIM_OPTS,     index=0, key="t1_stim")
        eating   = st.radio("🍽️ Eating today",               EAT_OPTS,      index=0, key="t1_eat")
    with col_b:
        sensory  = st.radio("🔊 Sensory sensitivity",        SENSORY_OPTS,  index=0, key="t1_sensory")
        routine  = st.radio("📅 Routine disruption",         ROUTINE_OPTS,  index=0, key="t1_routine")
        meltdown = st.radio("🌋 Meltdown / shutdown signs",  MELTDOWN_OPTS, index=0, key="t1_meltdown")
        new_beh  = st.radio("🆕 New or unusual behaviour",   NEWBEH_OPTS,   index=0, key="t1_newbeh")

    if st.button("Analyse Checklist 🔍", type="primary", key="cl_btn"):
        score = score_checklist(sleep, comm, stim, eating, sensory, routine, meltdown, new_beh)
        pred  = 0 if score <= 4 else (1 if score <= 13 else 2)
        headline, sub, tips, css = result_label(pred)

        mimi_col, res_col = st.columns([1, 3])
        with mimi_col:
            svg = MIMI_HAPPY_SVG if pred == 0 else (MIMI_CALM_SVG if pred == 1 else MIMI_ALERT_SVG)
            st.markdown(svg, unsafe_allow_html=True)
        with res_col:
            st.markdown(f'<div class="{css}"><strong>{headline}</strong><br>{sub}</div>', unsafe_allow_html=True)
            st.info(f"Checklist score: {score}")
            st.markdown("**Recommendations:**")
            st.text(tips)
            st.caption(DISCLAIMER)

    st.divider()
    st.markdown("### 💬 Ask Mimi a follow-up question")
    render_chatbot("cl")


# ── TAB 2: PHOTO CV ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("**Haar Cascade face detection → brightness → contrast → symmetry. No ML model required.**")
    uploaded = st.file_uploader("Upload photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="photo_up")

    if st.button("Analyse Photo 🔍", type="primary", key="photo_btn"):
        if uploaded is None:
            st.warning("Please upload a photo first.")
        else:
            pil_img = Image.open(uploaded).convert("RGB")
            results, annotated = analyse_photo(pil_img)
            pred = 0 if results["cv_score"] <= 1 else (1 if results["cv_score"] <= 3 else 2)

            mimi_col, img_col = st.columns([1, 2])
            with mimi_col:
                svg = MIMI_HAPPY_SVG if pred == 0 else (MIMI_CALM_SVG if pred == 1 else MIMI_ALERT_SVG)
                st.markdown(svg, unsafe_allow_html=True)
            with img_col:
                st.image(annotated, caption="Annotated output", use_container_width=True)

            if results["face_detected"]:
                css = "result-green" if pred == 0 else ("result-yellow" if pred == 1 else "result-red")
                st.markdown(f'<div class="{css}"><strong>✅ Face detected — {results["face_count"]} face(s)</strong></div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Brightness", f"{results['brightness']}/255")
                c2.metric("Contrast",   str(results["contrast"]))
                c3.metric("Symmetry diff", str(results["symmetry_score"]))
                st.markdown("**Observations:**")
                for o in results["observations"]:
                    st.write(f"• {o}")
            else:
                st.error("❌ No face detected in photo.")
                for o in results["observations"]:
                    st.write(f"• {o}")
            st.caption(DISCLAIMER)

    st.divider()
    st.markdown("### 💬 Ask Mimi a follow-up question")
    render_chatbot("photo")


# ── TAB 3: VIDEO CV ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("**Sample every 15th frame → Haar Cascade → aggregate stats. Recommended: MP4 under 30 s.**")
    video_file = st.file_uploader("Upload video (MP4)", type=["mp4", "avi", "mov"], key="video_up")

    if st.button("Analyse Video 🎥", type="primary", key="video_btn"):
        if video_file is None:
            st.warning("Please upload a video first.")
        else:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Analysing video frames..."):
                    summary, frame_stats, sample_frames = analyse_video(tmp_path, sample_every=15, max_frames=60)
                pred = 0 if summary["cv_score"] <= 1 else (1 if summary["cv_score"] <= 3 else 2)

                mimi_col, info_col = st.columns([1, 3])
                with mimi_col:
                    svg = MIMI_HAPPY_SVG if pred == 0 else (MIMI_CALM_SVG if pred == 1 else MIMI_ALERT_SVG)
                    st.markdown(svg, unsafe_allow_html=True)
                with info_col:
                    for o in summary["observations"]:
                        st.write(f"• {o}")
                    if summary["avg_brightness"] is not None:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Avg Brightness", f"{summary['avg_brightness']}/255")
                        c2.metric("Avg Contrast",   str(summary["avg_contrast"]))
                        c3.metric("Avg Symmetry",   str(summary["avg_symmetry"]))
                        c4.metric("Frames w/ face", f"{summary['frames_with_face']}/{summary['frames_sampled']}")

                if sample_frames:
                    st.markdown("**Sample frames:**")
                    cols = st.columns(min(len(sample_frames), 3))
                    for i, frm in enumerate(sample_frames[:3]):
                        cols[i].image(frm, use_container_width=True)

                b_data = [(s["time_s"], s["brightness"]) for s in frame_stats if s["brightness"] is not None]
                if b_data:
                    df = pd.DataFrame(b_data, columns=["Time (s)", "Brightness"])
                    st.markdown("**Brightness over time:**")
                    st.line_chart(df.set_index("Time (s)"))

                st.caption(DISCLAIMER)
            finally:
                os.unlink(tmp_path)

    st.divider()
    st.markdown("### 💬 Ask Mimi a follow-up question")
    render_chatbot("video")


# ── TAB 4: COMBINED ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("**Run checklist + photo CV + video CV together for a combined stress score.**")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Behaviour Checklist**")
        cb_sleep    = st.radio("🌙 Sleep",         SLEEP_OPTS,    index=0, key="cb_sleep")
        cb_comm     = st.radio("💬 Communication",  COMM_OPTS,     index=0, key="cb_comm")
        cb_stim     = st.radio("🔄 Stimming",       STIM_OPTS,     index=0, key="cb_stim")
        cb_eat      = st.radio("🍽️ Eating",         EAT_OPTS,      index=0, key="cb_eat")
        cb_sensory  = st.radio("🔊 Sensory",        SENSORY_OPTS,  index=0, key="cb_sensory")
        cb_routine  = st.radio("📅 Routine",        ROUTINE_OPTS,  index=0, key="cb_routine")
        cb_meltdown = st.radio("🌋 Meltdown signs", MELTDOWN_OPTS, index=0, key="cb_melt")
        cb_newbeh   = st.radio("🆕 New behaviour",  NEWBEH_OPTS,   index=0, key="cb_newbeh")

    with col_right:
        st.markdown("**Optional: Photo & Video**")
        cb_photo_up = st.file_uploader("Upload photo (optional)", type=["jpg","jpeg","png"], key="cb_photo")
        cb_video_up = st.file_uploader("Upload video (optional)", type=["mp4","avi","mov"],  key="cb_video")

    if st.button("Run Combined Analysis 🔍", type="primary", key="cb_btn"):
        cl_score = score_checklist(cb_sleep, cb_comm, cb_stim, cb_eat, cb_sensory, cb_routine, cb_meltdown, cb_newbeh)
        cv_photo = cv_video = 0
        photo_note = video_note = ""

        if cb_photo_up:
            pil_img = Image.open(cb_photo_up).convert("RGB")
            p, _ = analyse_photo(pil_img)
            cv_photo = p.get("cv_score", 0)
            if p["face_detected"]:
                photo_note = f"📷 brightness={p['brightness']}, contrast={p['contrast']}, symmetry={p['symmetry_score']}"

        if cb_video_up:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(cb_video_up.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Analysing video..."):
                    sv, _, _ = analyse_video(tmp_path, sample_every=15, max_frames=60)
                cv_video = sv.get("cv_score", 0)
                if sv["avg_brightness"] is not None:
                    video_note = f"🎥 avg brightness={sv['avg_brightness']}, frames w/ face={sv['frames_with_face']}"
            finally:
                os.unlink(tmp_path)

        total = cl_score + cv_photo + cv_video
        pred  = 0 if total <= 4 else (1 if total <= 13 else 2)
        headline, sub, tips, css = result_label(pred)

        mimi_col, res_col = st.columns([1, 3])
        with mimi_col:
            svg = MIMI_HAPPY_SVG if pred == 0 else (MIMI_CALM_SVG if pred == 1 else MIMI_ALERT_SVG)
            st.markdown(svg, unsafe_allow_html=True)
        with res_col:
            st.markdown(f'<div class="{css}"><strong>{headline}</strong><br>{sub}</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Checklist", cl_score)
            c2.metric("Photo CV",  cv_photo)
            c3.metric("Video CV",  cv_video)
            c4.metric("Total",     total)
            if photo_note:
                st.caption(photo_note)
            if video_note:
                st.caption(video_note)
            st.markdown("**Recommendations:**")
            st.text(tips)
            st.caption(DISCLAIMER)

    st.divider()
    st.markdown("### 💬 Ask Mimi a follow-up question")
    render_chatbot("combined")


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Mimi AI Caregiver — CV uses OpenCV only (Haar Cascade + NumPy). No external ML models.")
