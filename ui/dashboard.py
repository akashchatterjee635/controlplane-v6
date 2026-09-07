"""ControlPlane v6 — Streamlit Human Review Dashboard.

Provides a visual interface for:
- Submitting queries and seeing adaptive routing in action
- Reviewing flagged responses (approve / redact / deny)
- Monitoring metrics (cost, latency, escalation rates)

Usage:
    streamlit run ui/dashboard.py
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
import subprocess
import time
import socket

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Auto-start FastAPI backend if running locally or on Streamlit Cloud
if not is_port_in_use(8000) and os.getenv("API_BASE_URL") is None:
    # Ensure seed data is present
    if not os.path.exists("chroma_db"):
        print("Seeding knowledge base...")
        subprocess.run(["python", "data/knowledge_base/seed_data.py"], check=False)
        
    print("Starting FastAPI backend in the background...")
    subprocess.Popen(["uvicorn", "app.api:app", "--port", "8000", "--host", "0.0.0.0"])
    # Give it a few seconds to start
    time.sleep(3)

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ControlPlane v6 — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ ControlPlane v6")
st.sidebar.markdown("**Adaptive RAG** with Risk-Aware Routing")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["💬 Query Interface", "📋 Review Queue", "📊 Metrics"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(f"API: `{API_BASE}`")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def api_get(path: str) -> dict | list | None:
    """GET request to the ControlPlane API."""
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the ControlPlane API. Is the server running?")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(path: str, data: dict) -> dict | None:
    """POST request to the ControlPlane API."""
    try:
        resp = requests.post(f"{API_BASE}{path}", json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the ControlPlane API. Is the server running?")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Page: Query Interface
# ---------------------------------------------------------------------------
if page == "💬 Query Interface":
    st.title("💬 Query Interface")
    st.markdown("Submit a query and see how ControlPlane routes and processes it.")

    with st.form("query_form"):
        use_case = st.selectbox(
            "Select Policy Profile:",
            ["default", "customer_support", "internal_knowledge", "decision_support"],
            index=0,
            help="Determines routing thresholds, PII policy, and review rules."
        )
        
        query = st.text_area(
            "Enter your query:",
            placeholder="e.g., Compare the benefits of Kubernetes vs Docker Swarm for production deployments.",
            height=100,
        )
        submitted = st.form_submit_button("🚀 Submit Query", use_container_width=True)

    if submitted and query:
        with st.spinner("Processing query through ControlPlane..."):
            result = api_post("/api/v2/query", {"query": query, "use_case": use_case})

        if result:
            # Status badge
            status_val = result.get("status", "unknown")
            decision_val = result.get("decision", "unknown").upper()
            
            if status_val == "complete":
                if decision_val == "ALLOW":
                    st.success(f"✅ {decision_val} — {result.get('decision_reasoning', '')}")
                elif decision_val == "EDIT":
                    st.info(f"📝 {decision_val} — {result.get('decision_reasoning', '')}")
                elif decision_val == "FLAG":
                    st.warning(f"⚠️ {decision_val} — {result.get('decision_reasoning', '')}")
                elif decision_val == "BLOCK":
                    st.error(f"🛑 {decision_val} — {result.get('decision_reasoning', '')}")
                else:
                    st.success(f"✅ {decision_val}")
            elif status_val == "pending_review":
                st.warning("⏸️ REVIEW — Response flagged for human review")
            else:
                st.info(f"Status: {status_val}")
                
            if result.get("risk_labels"):
                st.markdown("**Risk Labels:** " + ", ".join([f"`{l}`" for l in result.get("risk_labels")]))

            # Routing info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                route = result.get("route", "unknown")
                st.metric("Route", route.upper(), delta=None)
            with col2:
                st.metric("Complexity", result.get("complexity_score", 0))
            with col3:
                st.metric("Risk Score", result.get("risk_score", 0))
            with col4:
                cost = result.get("cost", {})
                st.metric(
                    "Cost",
                    f"${cost.get('estimated_cost_usd', 0):.6f}",
                )

            # Response
            st.subheader("Response")
            st.markdown(result.get("response", "*No response generated*"))

            # Thread ID
            st.caption(f"Thread ID: `{result.get('thread_id', 'N/A')}`")

            # Audit log
            with st.expander("📜 Audit Log", expanded=False):
                for entry in result.get("audit_log", []):
                    st.text(entry)


# ---------------------------------------------------------------------------
# Page: Review Queue
# ---------------------------------------------------------------------------
elif page == "📋 Review Queue":
    st.title("📋 Human Review Queue")
    st.markdown("Responses flagged by the validation pipeline for human review.")

    if st.button("🔄 Refresh Queue", use_container_width=True):
        st.rerun()

    pending = api_get("/api/v1/pending-reviews")

    if pending is None:
        st.info("Unable to load reviews. Check API connection.")
    elif len(pending) == 0:
        st.success("✨ No pending reviews! All responses passed validation.")
    else:
        st.warning(f"⚠️ {len(pending)} response(s) awaiting review")

        for item in pending:
            thread_id = item.get("thread_id", "unknown")
            use_case = item.get("use_case", "default")

            with st.expander(
                f"🔎 Thread: {thread_id[:12]}... | Use Case: {use_case} | Risk: {item.get('risk_score', 0)} | "
                f"Confidence: {item.get('confidence', 0):.2f}",
                expanded=True,
            ):
                # Query and response
                st.markdown("**Original Query:**")
                st.info(item.get("query", ""))

                st.markdown("**Generated Response:**")
                st.warning(item.get("generated_response", ""))

                # Flags
                flags = item.get("validation_flags", [])
                if flags:
                    st.markdown("**Validation Flags:**")
                    for flag in flags:
                        st.error(f"🚩 {flag}")
                        
                reasoning = item.get("decision_reasoning", "")
                if reasoning:
                    st.markdown(f"**Reason for Review:** {reasoning}")

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", item.get("risk_score", 0))
                with col2:
                    st.metric("Complexity", item.get("complexity_score", 0))
                with col3:
                    st.metric("Confidence", f"{item.get('confidence', 0):.2f}")

                st.divider()

                # Review form
                st.markdown("**Your Decision:**")
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    decision = st.radio(
                        "Decision",
                        ["approve", "redact", "deny"],
                        key=f"decision_{thread_id}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )

                    redacted = None
                    if decision == "redact":
                        redacted = st.text_area(
                            "Redacted response:",
                            value=item.get("generated_response", ""),
                            key=f"redact_{thread_id}",
                        )

                    reason = st.text_input(
                        "Reason (optional):",
                        key=f"reason_{thread_id}",
                    )

                with col_b:
                    reviewer = st.text_input(
                        "Reviewer name:",
                        value="dashboard_user",
                        key=f"reviewer_{thread_id}",
                    )

                if st.button(
                    f"✅ Submit Review for {thread_id[:8]}...",
                    key=f"submit_{thread_id}",
                    use_container_width=True,
                ):
                    review_data = {
                        "decision": decision,
                        "reason": reason,
                        "reviewer": reviewer,
                    }
                    if redacted:
                        review_data["redacted_response"] = redacted

                    with st.spinner("Submitting review..."):
                        result = api_post(
                            f"/api/v1/review/{thread_id}",
                            review_data,
                        )

                    if result:
                        st.success(
                            f"✅ Review submitted: **{decision.upper()}**. "
                            f"Final response: {result.get('response', 'N/A')[:100]}..."
                        )
                        st.rerun()


# ---------------------------------------------------------------------------
# Page: Metrics & Feedback
# ---------------------------------------------------------------------------
elif page == "📊 Metrics":
    st.title("📈 Feedback & Tuning")
    st.markdown("Analyze human override rates and tune policy thresholds.")

    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from app.utils.feedback import compute_override_rate, suggest_threshold_adjustments
    from app.utils.audit import load_audit_log

    # Health check
    health = api_get("/health")
    if health:
        st.success(f"API Status: **{health.get('status', 'unknown')}** | Version: {health.get('version', 'N/A')}")
    else:
        st.error("API is not reachable")
        
    st.divider()
    
    st.subheader("Human Override Rates by Profile")
    st.write("An override occurs when a reviewer denies or modifies a flagged response.")
    
    profiles = ["default", "customer_support", "internal_knowledge", "decision_support"]
    
    for p in profiles:
        stats = compute_override_rate(p)
        if stats["total_reviews"] > 0:
            with st.expander(f"Profile: **{p}** ({stats['total_reviews']} reviews)", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews", stats["total_reviews"])
                col2.metric("Overrides", stats["overrides"])
                col3.metric("Override Rate", f"{stats['override_rate']:.1%}")
                
                suggestions = suggest_threshold_adjustments(p)
                for sug in suggestions:
                    if "Consider RAISE" in sug:
                        st.error(sug)
                    elif "Consider LOWER" in sug:
                        st.warning(sug)
                    else:
                        st.info(sug)
        else:
            with st.expander(f"Profile: **{p}** (0 reviews)", expanded=False):
                st.write("No review data available for this profile yet.")
                
    st.divider()
    st.subheader("Audit Log Browser")
    
    records = load_audit_log()
    if records:
        st.write(f"Showing last 100 audit records (Total: {len(records)})")
        
        # Display as a dataframe
        import pandas as pd
        
        df_records = []
        for r in reversed(records[-100:]): # Most recent first
            df_records.append({
                "query_id": r.get("query_id", "")[:8],
                "use_case": r.get("use_case", ""),
                "route": r.get("route", ""),
                "decision": r.get("decision", ""),
                "labels": ", ".join(r.get("risk_labels", [])),
                "action": r.get("reviewer_action", ""),
                "cost": f"${r.get('cost_usd', 0):.5f}",
                "latency": f"{r.get('latency_ms', 0):.0f}ms"
            })
            
        df = pd.DataFrame(df_records)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No audit records found. Run some queries to populate the log.")
