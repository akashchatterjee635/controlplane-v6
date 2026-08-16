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
        query = st.text_area(
            "Enter your query:",
            placeholder="e.g., Compare the benefits of Kubernetes vs Docker Swarm for production deployments.",
            height=100,
        )
        submitted = st.form_submit_button("🚀 Submit Query", use_container_width=True)

    if submitted and query:
        with st.spinner("Processing query through ControlPlane..."):
            result = api_post("/api/v1/query", {"query": query})

        if result:
            # Status badge
            status_val = result.get("status", "unknown")
            if status_val == "complete":
                st.success("✅ Query completed successfully")
            elif status_val == "pending_review":
                st.warning("⏸️ Response flagged for human review")
            else:
                st.info(f"Status: {status_val}")

            # Routing info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                route = result.get("route", "unknown")
                st.metric("Route", route.upper(), delta=None)
            with col2:
                st.metric("Complexity", result.get("complexity_score", 0))
            with col3:
                st.metric("Risk", result.get("risk_score", 0))
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

            with st.expander(
                f"🔎 Thread: {thread_id[:12]}... | Risk: {item.get('risk_score', 0)} | "
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
# Page: Metrics
# ---------------------------------------------------------------------------
elif page == "📊 Metrics":
    st.title("📊 System Metrics")
    st.markdown("Aggregate performance and cost metrics for ControlPlane.")

    # Health check
    health = api_get("/health")
    if health:
        st.success(f"API Status: **{health.get('status', 'unknown')}** | Version: {health.get('version', 'N/A')}")
    else:
        st.error("API is not reachable")

    st.divider()

    st.subheader("Pipeline Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🏎️ Fast Path
        - Low complexity + low risk
        - `Retrieve → Generate → Basic Validate`
        - Minimal overhead
        """)
    with col2:
        st.markdown("""
        ### 🔍 Verified Path
        - High complexity or risk
        - `Retrieve → Grade → Generate → Full Validate`
        - LLM-based document grading + output validation
        """)
    with col3:
        st.markdown("""
        ### 🛡️ HITL Review
        - Triggered when validation fails or confidence < threshold
        - Human reviewer: Approve / Redact / Deny
        - Full audit trail
        """)

    st.divider()

    # Pending reviews count
    pending = api_get("/api/v1/pending-reviews")
    if pending is not None:
        st.metric("Pending Reviews", len(pending))

    st.info(
        "💡 Detailed cost/latency metrics will be available in Phase 4 "
        "(evaluation framework). For now, per-query metrics are shown "
        "in the Query Interface tab."
    )
