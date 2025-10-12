"""
Streamlit dashboard for KG Reasoning system.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import json
import sys
from pathlib import Path


# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now you can import utils
from utils.config import Config
from utils.logging import setup_logging, get_logger


# Configure page
st.set_page_config(
    page_title="KG Reasoning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
setup_logging()
logger = get_logger('ui.streamlit')

# Configuration
config = Config()
API_BASE_URL = f"http://127.0.0.1:{config.api.port}"


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def make_prediction_request(head_entity: str, relation: str, top_k: int = 10, include_reasoning: bool = False):
    """Make a prediction request to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict/link",
            json={
                "head_entity": head_entity,
                "relation": relation,
                "top_k": top_k,
                "include_reasoning": include_reasoning
            },
            timeout=30
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status {response.status_code}"}

    except Exception as e:
        return False, {"error": str(e)}


def get_graph_stats():
    """Get graph statistics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/graph/stats", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


def main():
    # Title and header
    st.title("🧠 Knowledge Graph Reasoning System")
    st.markdown("**Graph Neural Network-based Link Prediction and Reasoning**")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # API Status
        st.subheader("API Status")
        is_healthy, health_data = check_api_health()

        if is_healthy:
            st.success("✅ API is running")
            if "model_loaded" in health_data:
                model_status = "✅ Loaded" if health_data["model_loaded"] else "❌ Not loaded"
                st.info(f"Model: {model_status}")
                # Show diagnostics when model not loaded
                if not health_data["model_loaded"]:
                    attempted = health_data.get('model_path_attempted')
                    load_err = health_data.get('model_load_error')
                    if attempted:
                        st.write(f"Attempted model path: `{attempted}`")
                    if load_err:
                        st.error(f"Model load error: {load_err}")
            if "database_connected" in health_data:
                db_status = "✅ Connected" if health_data["database_connected"] else "❌ Disconnected"
                st.info(f"Database: {db_status}")
        else:
            st.error("❌ API is not available")
            st.error(f"Error: {health_data.get('error', 'Unknown error')}")
            st.stop()

        # Graph Statistics
        st.subheader("📊 Graph Statistics")
        stats_success, graph_stats = get_graph_stats()

        if stats_success:
            st.metric("Entities", graph_stats.get("num_entities", 0))
            st.metric("Relationships", graph_stats.get("num_relationships", 0))
            st.metric("Relation Types", graph_stats.get("num_relation_types", 0))
        else:
            st.warning("Could not load graph statistics")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🔮 Link Prediction")

        # Input form
        with st.form("prediction_form"):
            head_entity = st.text_input(
                "Head Entity", 
                value="Albert_Einstein",
                help="Enter the name of the head entity"
            )

            relation = st.text_input(
                "Relation", 
                value="born_in",
                help="Enter the relation type"
            )

            col_k, col_reasoning = st.columns(2)
            with col_k:
                top_k = st.slider("Top K Predictions", min_value=1, max_value=20, value=10)

            with col_reasoning:
                include_reasoning = st.checkbox("Include Reasoning Paths", value=False)

            submitted = st.form_submit_button("🚀 Predict", type="primary")

        # Make prediction
        if submitted:
            if not head_entity or not relation:
                st.error("Please enter both head entity and relation")
            else:
                with st.spinner("Making prediction..."):
                    success, result = make_prediction_request(
                        head_entity, relation, top_k, include_reasoning
                    )

                if success:
                    st.success(f"✅ Prediction completed in {result['processing_time_ms']:.0f}ms")

                    # Display predictions
                    st.subheader("📋 Prediction Results")

                    predictions_df = pd.DataFrame(result['predictions'])
                    if not predictions_df.empty:
                        # Format the dataframe
                        predictions_df['score'] = predictions_df['score'].round(4)
                        predictions_df['confidence'] = (predictions_df['score'] - predictions_df['score'].min()) / (predictions_df['score'].max() - predictions_df['score'].min())

                        # Display as table
                        st.dataframe(
                            predictions_df[['rank', 'entity', 'score']],
                            use_container_width=True,
                            hide_index=True
                        )

                        # Visualization
                        fig = px.bar(
                            predictions_df.head(10), 
                            x='entity', 
                            y='score',
                            title=f"Top {min(10, len(predictions_df))} Predictions for ({head_entity}, {relation}, ?)",
                            labels={'score': 'Prediction Score', 'entity': 'Predicted Entity'}
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                    # Model confidence
                    confidence = result.get('model_confidence', 0)
                    st.metric("Model Confidence", f"{confidence:.3f}")

                    # Reasoning paths
                    if include_reasoning and result.get('reasoning_paths'):
                        st.subheader("🛤️ Reasoning Paths")
                        reasoning_data = result['reasoning_paths']

                        if 'paths' in reasoning_data and reasoning_data['paths']:
                            for i, path in enumerate(reasoning_data['paths'][:3]):
                                st.write(f"**Path {i+1}:** {' → '.join(path.get('path', []))}")
                                if 'explanation' in path:
                                    st.info(path['explanation'])
                        else:
                            st.info("No reasoning paths available")

                else:
                    st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

    with col2:
        st.header("ℹ️ System Information")

        # Sample entities and relations
        st.subheader("📝 Sample Data")
        st.write("**Sample Entities:**")
        sample_entities = [
            "Albert_Einstein", "Marie_Curie", "Isaac_Newton",
            "Germany", "Poland", "England", "Europe",
            "Nobel_Prize", "Theory_of_Relativity"
        ]
        for entity in sample_entities[:6]:
            st.code(entity)

        st.write("**Sample Relations:**")
        sample_relations = [
            "born_in", "located_in", "won", 
            "developed", "discovered", "is_a"
        ]
        for relation in sample_relations:
            st.code(relation)

        # Usage examples
        st.subheader("💡 Example Queries")
        examples = [
            ("Albert_Einstein", "born_in"),
            ("Marie_Curie", "won"),
            ("Germany", "located_in"),
            ("Theory_of_Relativity", "is_a")
        ]

        for head, rel in examples:
            if st.button(f"{head} → {rel}", key=f"example_{head}_{rel}"):
                # Update the form values (this is a simplified approach)
                st.session_state.head_entity = head
                st.session_state.relation = rel
                st.rerun()

        # API Documentation
        st.subheader("📚 API Documentation")
        st.markdown(f"[OpenAPI Docs]({API_BASE_URL}/docs)")
        st.markdown(f"[ReDoc]({API_BASE_URL}/redoc)")


if __name__ == "__main__":
    main()
