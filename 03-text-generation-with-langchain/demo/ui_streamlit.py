import json, time, textwrap, requests, streamlit as st, os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

st.set_page_config(
    page_title="Paper-to-Script 🎬",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container{padding-top:1.5rem}
    .stTabs [data-baseweb=tab]{font-weight:600;padding:8px 18px}

    .result-box{
      background:var(--secondary-background);
      border-left:6px solid var(--primary-color);
      padding:1rem;border-radius:.5rem;margin-top:.8rem;max-height:28rem;overflow-y:auto
    }
    .paper-card{
      background:var(--secondary-background);
      padding:.8rem 1rem;border:1px solid var(--primary-color);
      border-radius:.5rem;margin-bottom:.8rem
    }
    code{white-space:pre-wrap!important}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style='text-align:center;margin-bottom:0'>
      🎓 <span style='color:var(--primary-color)'>Paper-to-Script</span> Demo 🎬
    </h1>
    <p style='text-align:center'>
      Generate summaries or short scripts from scientific articles (arXiv).
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")

    api_url = st.text_input(
        "MLflow endpoint URL (`/invocations`)",
        value="https://localhost:5000/invocations",
    )

    st.markdown("---")
    st.subheader("Steps")
    col_a, col_b = st.columns(2)
    with col_a:
        do_extract  = st.checkbox("📥 Extract",  value=True)
        do_generate = st.checkbox("🎬 Script",   value=False)
    with col_b:
        do_analyze  = st.checkbox("🧐 Analyze",  value=True)

    st.markdown("---")
    st.subheader("Prompts")
    analysis_prompt = st.text_input(
        "Analysis prompt",
        value="Summarize the content in Portuguese (≈150 words).",
    )
    generation_prompt = st.text_input(
        "Initial script prompt (optional)",
        placeholder="e.g.: 'Create a 5-point script...' "
    )

    st.markdown("---")
    st.caption(
        "⚡ Tip: uncheck *Extract* to use cached PDFs, or check only *Extract* to download without analyzing."
    )

st.subheader("🔍 Search articles on arXiv")
query = st.text_input("Search term", value="graph neural networks")
cols = st.columns(3)
max_results   = cols[0].number_input("Number of articles",   1, 10, 3)
chunk_size    = cols[1].number_input("Chunk size",           200, 2000, 1200, step=100)
chunk_overlap = cols[2].number_input("Chunk overlap",          0,   800,  400, step=50)

if st.button("🚀 Run"):
    if not api_url.lower().startswith(("http://", "https://")):
        st.error("Invalid URL (must start with http:// or https://).")
        st.stop()

    payload = {
        "inputs": {
            "query":             [query],
            "max_results":       [max_results],
            "chunk_size":        [chunk_size],
            "chunk_overlap":     [chunk_overlap],
            "do_extract":        [do_extract],
            "do_analyze":        [do_analyze],
            "do_generate":       [do_generate],
            "analysis_prompt":   [analysis_prompt],
            "generation_prompt": [generation_prompt],
        },
        "params": {},
    }

    try:
        t0 = time.perf_counter()
        with st.spinner("Processing…"):
            resp = requests.post(api_url, json=payload, verify=False, timeout=600)
            resp.raise_for_status()
            result = resp.json()
        elapsed = time.perf_counter() - t0
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    rec = (
        result.get("predictions")
        or result.get("dataframe_records")
        or result.get("data")
        or result
    )[0]

    papers_json = rec.get("extracted_papers", "[]")
    main_output = rec.get("script", "")

    papers = json.loads(papers_json)

    t_art, t_sum, t_script = st.tabs(["📚 Articles", "📝 Summary", "🎬 Script"])

    with t_art:
        if not papers:
            st.info("No articles extracted or cached.")
        for idx, p in enumerate(papers, 1):
            with st.expander(f"{idx}. {p['title']}"):
                st.markdown(
                    f"<div class='paper-card'><code>"
                    f"{textwrap.shorten(p['text'], width=800, placeholder=' …')}"
                    f"</code></div>",
                    unsafe_allow_html=True,
                )

    with t_sum:
        if not do_analyze:
            st.info("Analysis is turned off.")
        elif do_analyze and do_generate:
            st.info(
                "When script generation is on, the service returns only the final script. "
                "Turn off *Script* if you only want the summary."
            )
        else:
            if main_output:
                st.markdown(f"<div class='result-box'>{main_output}</div>",
                            unsafe_allow_html=True)
            else:
                st.info("No summary returned.")

    with t_script:
        if not do_generate:
            st.info("Generation is turned off.")
        else:
            if main_output:
                st.markdown(f"<div class='result-box'>{main_output}</div>",
                            unsafe_allow_html=True)
            else:
                st.info("No script generated.")

    st.success(f"✓ Completed in {elapsed:.1f}s")
