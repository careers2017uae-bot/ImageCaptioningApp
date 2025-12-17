import streamlit as st
import os, time, uuid
import pandas as pd
from datetime import datetime
from io import BytesIO
from groq import Groq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Learning Intelligence Platform", layout="wide")

CSS = """

<style> .card{background:#ffffff;border-radius:14px;padding:16px;box-shadow:0 8px 18px rgba(0,0,0,.08);transition:.3s} .card:hover{transform:translateY(-6px);box-shadow:0 14px 30px rgba(0,0,0,.15)} </style>

"""
st.markdown(CSS, unsafe_allow_html=True)

def groq_client():
key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not key:
st.error("Groq API key missing")
st.stop()
return Groq(api_key=key)

def llm(prompt):
c = groq_client()
r = c.chat.completions.create(
model="llama3-70b-8192",
messages=[{"role":"user","content":prompt}],
temperature=0.4
)
return r.choices[0].message.content

def init_state():
defaults = {
"events": [],
"xp": 0,
"attempts": 0,
"correct": 0,
"start": time.time(),
"student_id": str(uuid.uuid4())[:8]
}
for k,v in defaults.items():
if k not in st.session_state:
st.session_state[k]=v

init_state()

def log_event(data):
st.session_state.events.append({"time":time.time(),**data})

def analytics_df():
return pd.DataFrame(st.session_state.events)

def pdf_report(title, text):
buf = BytesIO()
doc = SimpleDocTemplate(buf)
styles = getSampleStyleSheet()
story=[Paragraph(title,styles["Title"]),Spacer(1,12)]
for line in text.split("\n"):
story.append(Paragraph(line,styles["Normal"]))
story.append(Spacer(1,8))
doc.build(story)
buf.seek(0)
return buf

st.sidebar.title("Role")
role = st.sidebar.radio("Select",["Student","Teacher","School Admin"])

if role=="Student":
st.header("🎮 Student Learning Game")
grade = st.selectbox("Grade Level",["Primary","Secondary","Higher"])
mode = st.selectbox("Learning Mode",["🎮 Fun","🎯 Exam-oriented","🧠 Concept mastery"])
content = st.text_area("Paste learning content",height=180)
if st.button("Generate Game"):
    if not content:
        st.error("Content required")
    else:
        concepts = llm(f"Extract 5 key concepts from:\n{content}").split("\n")
        st.session_state["concepts"]=concepts

tabs = st.tabs(["🎮 Game","📊 My Analytics","🧠 AI Feedback"])

with tabs[0]:
    for c in st.session_state.get("concepts",[]):
        q = llm(f"Create one MCQ question on {c} and provide correct answer at end prefixed by ANSWER:")
        parts=q.split("ANSWER:")
        st.markdown(f"<div class='card'><b>{c}</b><br>{parts[0]}</div>",unsafe_allow_html=True)
        ans=st.text_input("Answer",key=c)
        if st.button("Submit",key=c+"b"):
            st.session_state.attempts+=1
            if parts[1].strip().lower() in ans.lower():
                st.success("Correct")
                st.session_state.correct+=1
                st.session_state.xp+=10
                log_event({"concept":c,"correct":1})
            else:
                st.error("Incorrect")
                log_event({"concept":c,"correct":0})

with tabs[1]:
    df=analytics_df()
    if not df.empty:
        st.metric("XP",st.session_state.xp)
        st.metric("Accuracy",f"{(st.session_state.correct/max(1,st.session_state.attempts))*100:.1f}%")
        st.bar_chart(df["concept"].value_counts())
        st.line_chart(df["correct"])

with tabs[2]:
    fb=llm(f"Give encouraging learning feedback based on: {st.session_state.events}")
    st.write(fb)
elif role=="Teacher":
st.header("📊 Teacher Analytics Dashboard")
df=analytics_df()
if df.empty:
st.info("No data yet")
else:
st.subheader("Concept Accuracy")
st.bar_chart(df.groupby("concept")["correct"].mean())
st.subheader("Engagement")
st.line_chart(df["correct"])
insight=llm(f"Provide teacher insights from analytics: {df.to_dict()}")
st.write(insight)
st.download_button("Download CSV",df.to_csv(index=False),"class_analytics.csv")

elif role=="School Admin":
st.header("🏫 School Admin Dashboard")
df=analytics_df()
if df.empty:
st.info("No platform data")
else:
st.metric("Active Students",1)
st.metric("Total Attempts",len(df))
st.area_chart(df["correct"])
insight=llm(f"Provide admin-level insights and curriculum gaps from: {df.to_dict()}")
st.write(insight)
pdf=pdf_report("School Analytics Report",insight)
st.download_button("Download PDF",pdf,"school_report.pdf")
