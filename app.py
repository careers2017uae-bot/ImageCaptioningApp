import os
import time
import uuid
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
from groq import Groq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="AI Learning Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# CUSTOM CSS
# ====================================================
st.markdown(
    """
    <style>
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        margin-bottom: 16px;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 32px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ====================================================
# GROQ LLM UTILITIES
# ====================================================
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("Groq API key not found. Set GROQ_API_KEY.")
        st.stop()
    return Groq(api_key=api_key)

def call_llm(prompt, temperature=0.4):
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# ====================================================
# SESSION STATE INITIALIZATION
# ====================================================
def init_state():
    defaults = {
        "student_id": str(uuid.uuid4())[:8],
        "events": [],
        "xp": 0,
        "attempts": 0,
        "correct": 0,
        "start_time": time.time(),
        "concepts": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ====================================================
# ANALYTICS UTILITIES
# ====================================================
def log_event(concept, correct):
    st.session_state.events.append({
        "timestamp": datetime.now(),
        "concept": concept,
        "correct": correct
    })

def get_analytics_df():
    if not st.session_state.events:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.events)

# ====================================================
# PDF REPORT
# ====================================================
def generate_pdf(title, text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12)
    ]
    for line in text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ====================================================
# SIDEBAR ROLE SELECTION
# ====================================================
st.sidebar.title("User Role")
role = st.sidebar.radio(
    "Select Role",
    ["Student", "Teacher", "School Admin"]
)

# ====================================================
# STUDENT VIEW
# ====================================================
if role == "Student":
    st.title("🎮 Gamified Learning")

    grade = st.selectbox("Grade Level", ["Primary", "Secondary", "Higher"])
    mode = st.selectbox("Learning Mode", ["🎮 Fun", "🎯 Exam-oriented", "🧠 Concept mastery"])

    content = st.text_area(
        "Paste learning content",
        height=200,
        placeholder="Paste any textbook, notes, or lesson content here..."
    )

    if st.button("Generate Learning Game"):
        if not content.strip():
            st.error("Please provide learning content.")
        else:
            with st.spinner("Analyzing content and creating game..."):
                raw = call_llm(
                    f"Extract 5 clear learning concepts from the following content:\n{content}"
                )
                st.session_state.concepts = [
                    c.strip("-• ").strip()
                    for c in raw.split("\n") if c.strip()
                ]
            st.success("Game generated successfully!")

    tabs = st.tabs(["🎮 Game", "📊 My Analytics", "🧠 AI Feedback"])

    # ---------------- GAME TAB ----------------
    with tabs[0]:
        for concept in st.session_state.concepts:
            question_block = call_llm(
                f"""
                Create ONE MCQ question to test understanding of:
                {concept}

                End with:
                ANSWER: <correct option>
                """
            )
            if "ANSWER:" not in question_block:
                continue

            question, answer = question_block.split("ANSWER:")
            st.markdown(f"<div class='card'><b>{concept}</b><br>{question}</div>", unsafe_allow_html=True)

            user_answer = st.text_input("Your Answer", key=concept)
            if st.button("Submit", key=f"btn_{concept}"):
                st.session_state.attempts += 1
                if answer.strip().lower() in user_answer.lower():
                    st.success("Correct!")
                    st.session_state.correct += 1
                    st.session_state.xp += 10
                    log_event(concept, 1)
                else:
                    st.error("Incorrect")
                    log_event(concept, 0)

    # ---------------- ANALYTICS TAB ----------------
    with tabs[1]:
        df = get_analytics_df()
        if df.empty:
            st.info("No analytics yet.")
        else:
            accuracy = (st.session_state.correct / max(1, st.session_state.attempts)) * 100
            st.metric("XP", st.session_state.xp)
            st.metric("Accuracy", f"{accuracy:.1f}%")
            st.bar_chart(df["concept"].value_counts())
            st.line_chart(df["correct"])

    # ---------------- FEEDBACK TAB ----------------
    with tabs[2]:
        if st.session_state.events:
            feedback = call_llm(
                f"Give supportive learning feedback based on this student data:\n{st.session_state.events}"
            )
            st.write(feedback)

# ====================================================
# TEACHER DASHBOARD
# ====================================================
elif role == "Teacher":
    st.title("📊 Teacher Analytics Dashboard")

    df = get_analytics_df()
    if df.empty:
        st.info("No student data available yet.")
    else:
        st.subheader("Concept-wise Accuracy")
        st.bar_chart(df.groupby("concept")["correct"].mean())

        st.subheader("Engagement Over Time")
        st.line_chart(df["correct"])

        insights = call_llm(
            f"Provide teaching insights and recommendations from this data:\n{df.to_dict()}"
        )
        st.write(insights)

        st.download_button(
            "Download CSV Report",
            df.to_csv(index=False),
            file_name="class_analytics.csv",
            mime="text/csv"
        )

# ====================================================
# SCHOOL ADMIN DASHBOARD
# ====================================================
elif role == "School Admin":
    st.title("🏫 School Admin Dashboard")

    df = get_analytics_df()
    if df.empty:
        st.info("No platform analytics available.")
    else:
        st.metric("Active Students", 1)
        st.metric("Total Attempts", len(df))

        st.area_chart(df["correct"])

        admin_insights = call_llm(
            f"Provide school-level insights, curriculum gaps, and recommendations:\n{df.to_dict()}"
        )
        st.write(admin_insights)

        pdf = generate_pdf("School Analytics Report", admin_insights)
        st.download_button(
            "Download PDF Report",
            pdf,
            file_name="school_analytics_report.pdf",
            mime="application/pdf"
        )
