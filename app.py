"""
Browser page for VirtueWeave.  Run it from the project folder with:

    streamlit run app.py
"""

import streamlit as st

from src.explainer import group_by_virtue
from src.pipeline import run_pipeline

st.set_page_config(page_title="VirtueWeave", page_icon="🧵")
st.title("VirtueWeave")
st.write("Describe your class in one sentence and get three practices that teach "
         "coding and build a virtue.")

sentence = st.text_area(
    "Your class",
    placeholder="Teaching loops to my 6-year-olds tomorrow, 15 minutes, they keep giving up.",
    max_chars=500,
)

if st.button("Find practices", type="primary"):
    with st.spinner("Thinking..."):
        try:
            result = run_pipeline(sentence)
        except Exception as error:  # the trace file records the crash too
            st.error(f"Something went wrong: {error}")
            st.stop()

    if not result.ok or not result.picks:
        st.warning(result.message or "No result.")
        st.stop()

    i = result.interpretation
    st.caption(
        f"Topic: {i.topic or 'not found'} · age: {i.age or '?'} · "
        f"minutes: {i.minutes or '?'} · read by the {i.source} parser"
    )
    if i.fallback_reason:
        st.caption(f"Fallback used: {i.fallback_reason}")
    if not result.explained:
        st.info("Showing the scoring reasons only (the written explanation was "
                "unavailable or did not pass the checks).")

    for virtue, group in group_by_virtue(result.picks).items():
        st.subheader(virtue.title())
        if result.explained:
            st.write(result.virtue_explanations[virtue])
        for pick in group:
            p = pick.practice
            with st.expander(
                f"{p['title']}  ·  {p['minutes']} min  ·  ages {p['age_min']}-{p['age_max']}",
                expanded=True,
            ):
                st.caption(f"{p['topic']} · score {pick.score:.2f}")
                if not result.explained:
                    st.markdown("\n".join(f"- {r}" for r in pick.reasons))
                st.markdown("**Sources:** " + "; ".join(e["label"] for e in pick.evidence))
    st.caption(f"Trace saved to {result.trace_path}")
