import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ── Questions ─────────────────────────────────────────────────
QUESTIONS = [
    {
        "key"    : "studytime",
        "ask"    : "📚 How many hours do you study per week?\n\n`1` = Less than 2 hrs | `2` = 2–5 hrs | `3` = 5–10 hrs | `4` = More than 10 hrs",
        "type"   : "int",
        "min"    : 1,
        "max"    : 4,
        "error"  : "Please enter a number between 1 and 4."
    },
    {
        "key"    : "absences",
        "ask"    : "🏫 How many days were you absent from school this year?",
        "type"   : "int",
        "min"    : 0,
        "max"    : 75,
        "error"  : "Please enter a number between 0 and 75."
    },
    {
        "key"    : "failures",
        "ask"    : "❌ How many classes have you failed before?\n\n`0` = None | `1` = One | `2` = Two | `3` = Three or more",
        "type"   : "int",
        "min"    : 0,
        "max"    : 3,
        "error"  : "Please enter a number between 0 and 3."
    },
    {
        "key"    : "higher",
        "ask"    : "🎯 Do you want to pursue higher education after school?\n\n`yes` or `no`",
        "type"   : "yn",
        "error"  : "Please type yes or no."
    },
    {
        "key"    : "internet",
        "ask"    : "🌐 Do you have internet access at home?\n\n`yes` or `no`",
        "type"   : "yn",
        "error"  : "Please type yes or no."
    },
    {
        "key"    : "famrel",
        "ask"    : "👨‍👩‍👧 How would you rate your family relationship quality?\n\n`1` = Very bad | `2` = Bad | `3` = Neutral | `4` = Good | `5` = Excellent",
        "type"   : "int",
        "min"    : 1,
        "max"    : 5,
        "error"  : "Please enter a number between 1 and 5."
    },
    {
        "key"    : "goout",
        "ask"    : "🎉 How often do you go out with friends?\n\n`1` = Very rarely | `2` = Rarely | `3` = Sometimes | `4` = Often | `5` = Very often",
        "type"   : "int",
        "min"    : 1,
        "max"    : 5,
        "error"  : "Please enter a number between 1 and 5."
    },
    {
        "key"    : "health",
        "ask"    : "💊 How would you rate your current health status?\n\n`1` = Very bad | `2` = Bad | `3` = Fair | `4` = Good | `5` = Very good",
        "type"   : "int",
        "min"    : 1,
        "max"    : 5,
        "error"  : "Please enter a number between 1 and 5."
    },
]

# ── Parse helpers ─────────────────────────────────────────────
def parse_yn(text: str):
    t = text.strip().lower()
    if t in ["yes", "y", "1", "yeah", "yep"]: return 1
    if t in ["no",  "n", "0", "nah", "nope"]: return 0
    return None

def parse_int(text: str, min_val: int, max_val: int):
    try:
        val = int(text.strip())
        if min_val <= val <= max_val:
            return val
        return None
    except ValueError:
        return None

def parse_answer(q: dict, text: str):
    if q["type"] == "yn":
        return parse_yn(text)
    return parse_int(text, q["min"], q["max"])

# ── SHAP chart ────────────────────────────────────────────────
def render_shap_chart(shap_values: dict):
    labels = list(shap_values.keys())
    values = list(shap_values.values())
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    sorted_pairs = sorted(zip(values, labels), key=lambda x: abs(x[0]))
    values_s = [p[0] for p in sorted_pairs]
    labels_s = [p[1] for p in sorted_pairs]
    colors_s = ["#2ecc71" if v > 0 else "#e74c3c" for v in values_s]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels_s, values_s, color=colors_s, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title("Why this grade was predicted", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

# ── Session state init ────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "step"         not in st.session_state: st.session_state.step         = 0
if "answers"      not in st.session_state: st.session_state.answers      = {}
if "done"         not in st.session_state: st.session_state.done         = False
if "started"      not in st.session_state: st.session_state.started      = False

# ── Header ────────────────────────────────────────────────────
st.title("🎓 Student Performance Predictor")
st.caption("Answer a few questions and I will predict your final grade.")
st.divider()

# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "chart":
            st.pyplot(msg["fig"])
        else:
            st.markdown(msg["content"])

# ── Start conversation ────────────────────────────────────────
if not st.session_state.started:
    intro = (
        "👋 Hi! I'm going to ask you **8 quick questions** about your study habits "
        "and lifestyle.\n\nBased on your answers, I'll predict your final grade "
        "and explain exactly why.\n\nLet's begin! 🚀"
    )
    with st.chat_message("assistant"):
        st.markdown(intro)
    st.session_state.messages.append({"role": "assistant", "content": intro})

    first_q = QUESTIONS[0]["ask"]
    with st.chat_message("assistant"):
        st.markdown(first_q)
    st.session_state.messages.append({"role": "assistant", "content": first_q})

    st.session_state.started = True
    st.rerun()

# ── Chat input ────────────────────────────────────────────────
if not st.session_state.done:
    user_input = st.chat_input("Type your answer here...")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        current_step = st.session_state.step
        current_q    = QUESTIONS[current_step]
        parsed       = parse_answer(current_q, user_input)

        if parsed is None:
            # Invalid input — ask again
            error_msg = f"⚠️ {current_q['error']}\n\n{current_q['ask']}"
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        else:
            # Valid — store answer
            st.session_state.answers[current_q["key"]] = parsed
            st.session_state.step += 1
            next_step = st.session_state.step

            if next_step < len(QUESTIONS):
                # Ask next question
                progress  = f"✅ Got it! ({next_step}/{len(QUESTIONS)})"
                next_q    = QUESTIONS[next_step]["ask"]

                with st.chat_message("assistant"):
                    st.markdown(progress)
                    st.markdown(next_q)

                st.session_state.messages.append({"role": "assistant", "content": progress})
                st.session_state.messages.append({"role": "assistant", "content": next_q})

            else:
                # All questions answered — call API
                with st.chat_message("assistant"):
                    st.markdown("✅ All done! Analysing your answers...")

                st.session_state.messages.append({
                    "role"   : "assistant",
                    "content": "✅ All done! Analysing your answers..."
                })

                try:
                    response = requests.post(
                        f"{BACKEND_URL}/predict",
                        json=st.session_state.answers,
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()

                    grade      = result["predicted_grade"]
                    confidence = result["confidence"]
                    shap_vals  = result["shap_values"]
                    advice     = result["advice"]

                    grade_emoji = {
                        "A": "🟢", "B": "🔵",
                        "C": "🟡", "D": "🟠", "F": "🔴"
                    }.get(grade, "⚪")

                    result_text = (
                        f"---\n"
                        f"### 🎓 Prediction Complete!\n\n"
                        f"**Predicted Grade:** {grade_emoji} **{grade}**\n\n"
                        f"**Confidence:** {round(confidence * 100, 1)}%\n\n"
                        f"💡 {advice}"
                    )

                    with st.chat_message("assistant"):
                        st.markdown(result_text)
                        st.markdown("**📊 Why this grade was predicted:**")
                        fig = render_shap_chart(shap_vals)
                        st.pyplot(fig)

                    st.session_state.messages.append({
                        "role"   : "assistant",
                        "content": result_text
                    })
                    st.session_state.messages.append({
                        "role" : "assistant",
                        "type" : "chart",
                        "fig"  : fig
                    })

                    st.session_state.done = True

                except requests.exceptions.ConnectionError:
                    err = "❌ Could not connect to the backend. Make sure FastAPI is running."
                    with st.chat_message("assistant"):
                        st.markdown(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

                except Exception as e:
                    err = f"❌ Something went wrong: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

        st.rerun()

# ── Restart button ────────────────────────────────────────────
if st.session_state.done:
    st.divider()
    if st.button("🔄 Start over"):
        for key in ["messages", "step", "answers", "done", "started"]:
            del st.session_state[key]
        st.rerun()