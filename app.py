import os
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# ----------------------
# State definition
# We'll pass around a Python dict with:
# - app_url
# - html
# - features
# - test_plan
# ----------------------

def fetch_html_node(state: dict) -> dict:
    """Fetch HTML from the provided app URL."""
    app_url = state["app_url"]
    headers = {"User-Agent": "Mozilla/5.0"}
    print(f"[fetch_html_node] Fetching HTML from: {app_url}")
    resp = requests.get(app_url, headers=headers)
    resp.raise_for_status()
    state["html"] = resp.text
    return state

def extract_features_node(state: dict) -> dict:
    """Parse HTML and extract features text from headings and feature classes."""
    html_content = state["html"]
    soup = BeautifulSoup(html_content, "html.parser")
    texts = []

    # Change selectors as needed for your app's DOM
    for tag in soup.find_all(['h2', 'h3', 'li', 'p']):
        # Collect headings or anything with 'feature' class
        if tag.name in ['h2', 'h3'] or ('feature' in (tag.get('class') or [])):
            text = tag.get_text(strip=True)
            if text:
                texts.append(text)

    features_text = "\n".join(sorted(set(texts)))
    print(f"[extract_features_node] Found {len(texts)} feature entries")
    state["features"] = features_text
    return state

def generate_test_plan_node(state: dict) -> dict:
    """Call Gemini to generate a QA Test Plan from extracted features."""
    if not state.get("features"):
        raise ValueError("No features extracted to analyse.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3
    )

    prompt = PromptTemplate(
        input_variables=["features"],
        template=(
            "You are an experienced QA test architect specializing in LLM-powered business applications. "
            "The input below is a list of application features extracted from HTML.\n\n"
            "For each feature: provide description, functional areas, business processes, KPIs, personas, "
            "data model, 2-3 test cases (positive, negative, edge), dependencies.\n\n"
            "Then provide overall recommended test strategy and risks.\n\n"
            "Output as JSON array with keys: feature_name, description, functional_areas, business_processes, "
            "kpis, personas, data_model, test_cases, dependencies, recommended_test_strategy, risks.\n\n"
            "Features:\n{features}"
        )
    )

    print(f"[generate_test_plan_node] Sending features to Gemini ({len(state['features'])} chars)...")
    chain = prompt | llm
    result = chain.invoke({"features": state["features"]})
    state["test_plan"] = result.content
    return state

# ----------------------
# Build the LangGraph workflow
# ----------------------
def build_graph():
    workflow = StateGraph(dict)  # our state will be a simple dict

    # Add our nodes
    workflow.add_node("fetch_html", fetch_html_node)
    workflow.add_node("extract_features", extract_features_node)
    workflow.add_node("generate_test_plan", generate_test_plan_node)

    # Define edges
    workflow.add_edge(START, "fetch_html")
    workflow.add_edge("fetch_html", "extract_features")
    workflow.add_edge("extract_features", "generate_test_plan")
    workflow.add_edge("generate_test_plan", END)

    return workflow.compile()

# ----------------------
# Main runner
# ----------------------
if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError("Please set GEMINI_API_KEY in your environment.")

    # Replace with your target app URL
    start_state = {"app_url": "https://deepthought.tigeranalyticstest.in/aidatamaestro/"}

    app = build_graph()
    result_state = app.invoke(start_state)

    print("\n=== GENERATED TEST PLAN ===")
    print(result_state["test_plan"])
