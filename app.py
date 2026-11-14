"""ButterflAI - students teach a confused AI classmate; the lecturer sees the gaps.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Works with no API key (offline demo mode). Add a Claude API key in the
sidebar to let Bruco read your real lecture slides and judge explanations.
"""
import json
import random
import re

import pandas as pd
import streamlit as st

COURSE = "Smart Factories"
LECTURE = "Lecture 5 - Data and maintenance in the smart factory"
DEFAULT_MODEL = "claude-sonnet-5-5"

# Placeholder concepts. Replace them with the concepts from your own lecture.
# "bruco" is what the confused classmate says; "keywords" drive offline scoring.
CONCEPTS = [
    {
        "name": "OEE",
        "misconception": "OEE is availability x performance; quality is not part of it.",
        "bruco": "So OEE is just availability times performance, right? If the "
                 "machine runs the whole shift at full speed, OEE is 100%, even "
                 "if some of the parts get scrapped.",
        "keywords": ["quality", "scrap", "defect", "good part", "three"],
    },
    {
        "name": "Digital twin",
        "misconception": "A digital twin is a 3D model or an offline simulation.",
        "bruco": "A digital twin is basically a detailed 3D CAD model of the "
                 "machine. Once you have drawn it, you have the twin.",
        "keywords": ["real-time", "real time", "sensor", "live", "data", "synchron", "update"],
    },
    {
        "name": "Predictive maintenance",
        "misconception": "Predictive maintenance means servicing on a fixed schedule.",
        "bruco": "Predictive maintenance is when you replace the bearing every "
                 "six months so it never fails. That's it, right?",
        "keywords": ["condition", "sensor", "data", "preventive", "schedule", "vibration", "remaining"],
    },
    {
        "name": "MES vs ERP",
        "misconception": "MES and ERP do the same job, so a factory needs only one.",
        "bruco": "MES and ERP are the same thing. Both plan orders and finances, "
                 "so a factory only needs one of them.",
        "keywords": ["shop floor", "shop-floor", "execution", "real-time", "real time", "business", "production"],
    },
    {
        "name": "Edge vs cloud",
        "misconception": "Edge computing sends all data to the cloud before processing.",
        "bruco": "Edge computing means every sensor reading goes to the cloud, "
                 "gets processed there, and the answer comes back to the machine.",
        "keywords": ["local", "near", "latency", "close", "bandwidth", "on the machine", "device"],
    },
    {
        "name": "Cyber-physical systems",
        "misconception": "A CPS is any machine with a computer attached.",
        "bruco": "A cyber-physical system is just any machine with a screen and "
                 "a PC next to it.",
        "keywords": ["sensor", "actuator", "feedback", "control", "loop", "network", "integrat"],
    },
]
# Share of the simulated class that catches each misconception.
CATCH_RATE = {"OEE": 0.70, "Digital twin": 0.45, "Predictive maintenance": 0.30,
              "MES vs ERP": 0.60, "Edge vs cloud": 0.50, "Cyber-physical systems": 0.65}
LABELS = {0: "Missed", 1: "Partly caught", 2: "Caught and explained"}
ASKS_FOR_ANSWER = re.compile(
    r"(tell|give|show) me (the )?(right |correct )?answer|what('s| is) the (right |correct )?answer"
    r"|just (tell|give)|i don'?t know", re.I)
REFUSAL = ("I can't tell you, I'm the one who's confused! You were in the same "
           "lecture. What part of what I said sounds wrong to you?")

BRUCO_RULES = """You are Bruco, a first-year classmate who misunderstood part of
today's lecture. You are friendly and sincerely confused. You are not a tutor.
Rules: never state or confirm the correct answer. If the student asks for the
answer, refuse in character. If their explanation is wrong or vague, ask one
short follow-up question that exposes the gap. Judge reasoning, not language."""


# ---------- data ----------
def simulated_class(n=40, seed=7):
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        for c in CONCEPTS:
            caught = rng.random() < CATCH_RATE.get(c["name"], 0.5)
            score = 2 if caught else rng.choice([0, 0, 1])
            rows.append({"student": f"Student {i:02d}", "concept": c["name"], "score": score})
    return pd.DataFrame(rows)


# ---------- LLM (optional) ----------
def ask_llm(system, user, max_tokens=700):
    import anthropic
    client = anthropic.Anthropic(api_key=st.session_state.api_key)
    msg = client.messages.create(
        model=st.session_state.model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return msg.content[0].text


def parse_json(text):
    match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    return json.loads(match.group(1))


def read_upload(file):
    if file is None:
        return ""
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
    return file.read().decode("utf-8", errors="ignore")


def concepts_from_slides(lecture_text):
    """Ask the model for three misconceptions grounded in the real slides."""
    prompt = ("From the lecture below, pick three key concepts. For each, write a "
              "plausible misconception a real student would have. Reply with JSON only: "
              '[{"name": "...", "misconception": "...", "bruco": "2-3 sentences in which '
              'Bruco states the misconception as if it were true"}]\n\nLECTURE:\n'
              + lecture_text[:15000])
    items = parse_json(ask_llm(BRUCO_RULES, prompt))
    return [{**item, "keywords": []} for item in items][:3]


def evaluate(concept, explanation, lecture_text):
    """Return (score 0-2, Bruco's reply). Falls back to keywords with no API."""
    if st.session_state.get("api_key"):
        try:
            prompt = (f"Concept: {concept['name']}\nYour misconception: {concept['misconception']}\n"
                      f"Student's explanation: {explanation}\n\n"
                      "Score the explanation: 0 = did not catch the error, 1 = caught it but "
                      "the explanation is incomplete, 2 = caught it and explained it correctly. "
                      'Reply with JSON only: {"score": 0, "reply": "Bruco in character; if the '
                      'score is below 2 ask one probing question, never give the answer"}\n\n'
                      "LECTURE:\n" + lecture_text[:15000])
            out = parse_json(ask_llm(BRUCO_RULES, prompt))
            return max(0, min(2, int(out["score"]))), out["reply"]
        except Exception as err:  # keep the demo alive if the API fails
            st.toast(f"API problem, using offline scoring ({type(err).__name__})")
    hits = sum(k in explanation.lower() for k in concept["keywords"])
    score = 2 if hits >= 2 else 1
    if hits == 0 and concept["keywords"]:
        score = 0
    reply = {2: "Ohh, now it makes sense. Thanks, I had that wrong!",
             1: "Hmm, I think I follow, but why exactly does that make my version wrong?",
             0: "I'm not sure I see the problem. Which part of what I said is wrong?"}[score]
    return score, reply


# ---------- student check-in ----------
def start_checkin(lecture_text):
    chosen = None
    if st.session_state.get("api_key") and lecture_text:
        try:
            chosen = concepts_from_slides(lecture_text)
        except Exception as err:
            st.toast(f"Could not read slides, using built-in concepts ({type(err).__name__})")
    if not chosen:
        chosen = random.sample(CONCEPTS, 3)
    st.session_state.chosen = chosen
    st.session_state.idx = 0
    st.session_state.attempt = 1
    st.session_state.results = []
    st.session_state.chat = [("assistant", chosen[0]["bruco"] + " Did I get that right?")]


def handle_answer(text, lecture_text):
    s = st.session_state
    s.chat.append(("user", text))
    if ASKS_FOR_ANSWER.search(text):
        s.chat.append(("assistant", REFUSAL))
        return
    concept = s.chosen[s.idx]
    score, reply = evaluate(concept, text, lecture_text)
    s.chat.append(("assistant", reply))
    if score < 2 and s.attempt == 1:
        s.attempt = 2
        return
    s.results.append({"concept": concept["name"], "score": score,
                      "misconception": concept["misconception"]})
    s.idx += 1
    s.attempt = 1
    if s.idx < len(s.chosen):
        s.chat.append(("assistant", "Next one. " + s.chosen[s.idx]["bruco"]
                       + " Did I get that right?"))
    else:
        live = pd.DataFrame([{"student": "You (live)", "concept": r["concept"],
                              "score": r["score"]} for r in s.results])
        s.checkins = pd.concat(
            [s.checkins[s.checkins.student != "You (live)"], live], ignore_index=True)


def student_view(lecture_text):
    s = st.session_state
    st.subheader("Teach Bruco")
    st.caption("Bruco misunderstood today's lecture. Find the mistakes and explain "
               "the concept. Bruco will never give you the answer.")
    if "chat" not in s:
        if st.button("Start today's check-in", type="primary"):
            start_checkin(lecture_text)
            st.rerun()
        return
    for role, text in s.chat:
        with st.chat_message(role, avatar="🐛" if role == "assistant" else None):
            st.write(text)
    if s.idx < len(s.chosen):
        with st.form("answer", clear_on_submit=True):
            text = st.text_area("Your explanation", height=100)
            if st.form_submit_button("Send") and text.strip():
                handle_answer(text.strip(), lecture_text)
                st.rerun()
        return
    st.success("Check-in complete. Your report was sent to the lecturer dashboard.")
    report = pd.DataFrame(s.results)
    report["result"] = report.score.map(LABELS)
    st.table(report[["concept", "result"]])
    weakest = report.sort_values("score").iloc[0]
    if weakest.score < 2:
        st.info(f"Revisit before the exam: **{weakest.concept}**. "
                f"Flag for the lecturer: \"{weakest.misconception}\"")
    if st.button("New check-in"):
        del s["chat"]
        st.rerun()


# ---------- lecturer dashboard ----------
def lecturer_view():
    df = st.session_state.checkins
    st.subheader(f"{COURSE} · {LECTURE}")
    st.caption("Class data is simulated for this demo. Live check-ins from the "
               "student tab are added as \"You (live)\".")
    by_concept = df.groupby("concept").score
    caught = (by_concept.apply(lambda x: (x == 2).mean()) * 100).round()
    weakest = caught.idxmin()

    c1, c2, c3 = st.columns(3)
    c1.metric("Students checked in", df.student.nunique())
    c2.metric("Misconceptions caught", f"{(df.score == 2).mean():.0%}")
    c3.metric("Weakest concept", weakest)

    st.markdown("**How the class did on each misconception (% of students)**")
    shares = (df.assign(result=df.score.map(LABELS))
                .pivot_table(index="concept", columns="result", values="student",
                             aggfunc="count", fill_value=0))
    shares = (shares.div(shares.sum(axis=1), axis=0) * 100).round()
    shares = shares.reindex(columns=[LABELS[0], LABELS[1], LABELS[2]], fill_value=0)
    st.bar_chart(shares.loc[caught.sort_values().index], horizontal=True,
                 color=["#d9534f", "#f0ad4e", "#5cb85c"])

    st.markdown("**Open the next lecture with these**")
    lookup = {c["name"]: c["misconception"] for c in CONCEPTS}
    for name in caught.sort_values().index[:2]:
        st.warning(f"**{name}** — only {caught[name]:.0f}% caught it. "
                   f"Common belief: \"{lookup.get(name, 'see live check-ins')}\"")

    left, right = st.columns(2)
    with left:
        st.markdown("**Students who may need support**")
        avg = df.groupby("student").score.mean().sort_values().head(5)
        st.dataframe(avg.rename("average score (0-2)").round(2))
    with right:
        st.markdown("**Personalised final exam**")
        student = st.selectbox("Student", sorted(df.student.unique()))
        mine = df[df.student == student]
        open_gaps = mine[mine.score < 2].concept.tolist()
        proven = mine[mine.score == 2].concept.tolist()
        st.write("Still to prove at the exam:", ", ".join(open_gaps) or "nothing")
        st.write("Already proven in check-ins:", ", ".join(proven) or "nothing yet")


# ---------- app ----------
def main():
    st.set_page_config(page_title="ButterflAI", page_icon="🦋", layout="wide")
    s = st.session_state
    if "checkins" not in s:
        s.checkins = simulated_class()

    with st.sidebar:
        st.title("🦋 ButterflAI")
        st.write("Students teach a confused AI classmate. The lecturer sees "
                 "what the class has not understood.")
        s.api_key = st.text_input("Claude API key (optional)", type="password")
        s.model = st.text_input("Model", DEFAULT_MODEL)
        upload = st.file_uploader("Lecture slides (PDF or TXT)", type=["pdf", "txt"])
        try:
            lecture_text = read_upload(upload)
        except Exception as err:
            lecture_text = ""
            st.error(f"Could not read the file ({type(err).__name__}).")
        mode = "Live: Bruco reads your slides" if s.api_key and lecture_text else (
            "Live scoring, built-in concepts" if s.api_key else "Offline demo mode")
        st.caption(f"Mode: {mode}")

    tab_student, tab_lecturer = st.tabs(["Student check-in", "Lecturer dashboard"])
    with tab_student:
        student_view(lecture_text)
    with tab_lecturer:
        lecturer_view()


main()
