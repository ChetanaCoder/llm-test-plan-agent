import os
from langchain_community.tools.requests import RequestsWrapper
from langchain_community.document_loaders import BSHTMLLoader
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import Graph

def fetch_html(app_url: str) -> str:
    req = RequestsWrapper()
    return req.get(app_url, headers={"User-Agent": "Mozilla/5.0"})

def extract_features(html_content: str) -> str:
    loader = BSHTMLLoader.from_html_string(html_content)
    soup = loader.soup
    texts = []
    for tag in soup.find_all(['h2','h3','li','p']):
        if tag.name in ['h2','h3'] or (tag.get('class') and 'feature' in tag.get('class')):
            texts.append(tag.get_text(strip=True))
    return "\n".join(set(texts))

def generate_test_plan(features_text: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3
    )
    prompt = PromptTemplate(
        input_variables=["features"],
        template=(
            "You are an experienced QA test architect specializing in LLM-powered business applications. "
            "The input below is a list of application features automatically extracted from HTML.\n\n"
            "For each feature: provide description, functional areas, business processes, KPIs, personas, "
            "data model, 2-3 test cases, dependencies.\n\n"
            "Then provide recommended test strategy and risks.\n\n"
            "Output as JSON array with keys: feature_name, description, functional_areas, business_processes, "
            "kpis, personas, data_model, test_cases, dependencies, recommended_test_strategy, risks.\n\n"
            "Features:\n{features}"
        )
    )
    result = (prompt | llm).invoke({"features": features_text})
    return {"test_plan": result.content}

graph = Graph()

@graph.node()
def start(state):
    state["html"] = fetch_html(state["app_url"])
    return state

@graph.node()
def parse(state):
    state["features"] = extract_features(state["html"])
    return state

@graph.node()
def analyse(state):
    state["test_plan"] = generate_test_plan(state["features"])
    return state

graph.add_edge("start", "parse")
graph.add_edge("parse", "analyse")

if __name__ == "__main__":
    start_state = {"app_url": "https://example.com"}
    final = graph.run(start_state)
    print(final["test_plan"])
