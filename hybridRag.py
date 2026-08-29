"""
Hybrid RAG: Vector RAG (Chroma) + Graph RAG (networkx)
--------------------------------------------------------
Extends the original rag.py by adding a knowledge-graph layer.

- Vector RAG (what you already had): good for single-hop fact lookup
- Graph RAG (new): extracts entities + relationships from your document
  using the LLM itself, builds a graph, and answers relational /
  multi-hop questions by traversing that graph.

A simple router decides which path to use per question.
"""

import json
import os
import pickle

import networkx as nx
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

DOCUMENT_PATH = "my_document.txt"
GRAPH_PATH = "knowledge_graph.pkl"
CHROMA_DIR = "./chroma_db"

# ---------------------------------------------------------------------
# STEP 1-4: Same as your original rag.py -- load, split, embed, store
# ---------------------------------------------------------------------
loader = TextLoader(DOCUMENT_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="llama3.2", temperature=0)

vector_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# ---------------------------------------------------------------------
# STEP 5 (NEW): Build the knowledge graph from the same chunks
# ---------------------------------------------------------------------
# We ask the LLM to extract (subject, relation, object) triples from
# each chunk. This is the "entity + relationship extraction" step that
# turns unstructured text into structured graph data.

EXTRACTION_PROMPT = """Extract entities and relationships from the text below.

Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{{"triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Rules:
- subject and object should be short entity names (people, places, organizations, concepts)
- relation should be a short verb phrase (e.g. "acquired", "works_for", "located_in")
- Extract only clear, explicit relationships stated in the text
- If no relationships are found, return {{"triples": []}}

Text:
{text}
"""


def extract_triples(text: str) -> list[dict]:
    """Ask the LLM to pull (subject, relation, object) triples out of a chunk."""
    prompt = EXTRACTION_PROMPT.format(text=text)
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Models sometimes wrap JSON in ```json fences despite instructions -- strip them.
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()

    try:
        data = json.loads(content)
        return data.get("triples", [])
    except json.JSONDecodeError:
        print(f"  (skipped one chunk -- could not parse JSON: {content[:80]}...)")
        return []


def build_graph(chunks) -> nx.DiGraph:
    """Build a directed graph: nodes = entities, edges = relationships."""
    graph = nx.DiGraph()
    print("\nExtracting entities and relationships for the knowledge graph...")

    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk.page_content)
        for t in triples:
            subj, rel, obj = t.get("subject"), t.get("relation"), t.get("object")
            if subj and rel and obj:
                graph.add_edge(subj.strip(), obj.strip(), relation=rel.strip())
        print(f"  chunk {i + 1}/{len(chunks)} -> {len(triples)} triples found")

    return graph


def load_or_build_graph(chunks) -> nx.DiGraph:
    """Reuse a saved graph if present, otherwise build (and save) a new one."""
    if os.path.exists(GRAPH_PATH):
        print(f"\nLoading existing knowledge graph from {GRAPH_PATH}")
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)

    graph = build_graph(chunks)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved knowledge graph ({graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges) to {GRAPH_PATH}")
    return graph


knowledge_graph = load_or_build_graph(chunks)

# ---------------------------------------------------------------------
# STEP 6 (NEW): Graph retrieval -- find relevant entities, traverse, answer
# ---------------------------------------------------------------------

ENTITY_EXTRACTION_PROMPT = """Identify the key entity names mentioned in this question.
Return ONLY a JSON list of strings, e.g. ["Entity One", "Entity Two"].
No explanation, no markdown.

Question: {question}
"""


def extract_query_entities(question: str) -> list[str]:
    prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []


def graph_context_for_entities(graph: nx.DiGraph, entities: list[str], hops: int = 2) -> str:
    """
    Traverse outward from each matched entity up to `hops` steps,
    collecting the relationships (edges) along the way as text facts.
    """
    facts = []
    visited_nodes = set()

    # Fuzzy match: find graph nodes that contain the entity text (case-insensitive)
    matched_nodes = []
    for entity in entities:
        for node in graph.nodes:
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                matched_nodes.append(node)

    matched_nodes = list(set(matched_nodes))

    for start_node in matched_nodes:
        frontier = {start_node}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                if node in visited_nodes:
                    continue
                visited_nodes.add(node)
                for _, target, data in graph.out_edges(node, data=True):
                    facts.append(f"{node} --[{data['relation']}]--> {target}")
                    next_frontier.add(target)
                for source, _, data in graph.in_edges(node, data=True):
                    facts.append(f"{source} --[{data['relation']}]--> {node}")
                    next_frontier.add(source)
            frontier = next_frontier

    return "\n".join(facts) if facts else ""


GRAPH_ANSWER_PROMPT = """Answer the question using ONLY the facts below.
If the facts don't fully answer the question, say what you can and note what's missing.

Facts (as entity relationships):
{facts}

Question: {question}
"""


def graph_qa(question: str) -> str:
    entities = extract_query_entities(question)
    facts = graph_context_for_entities(knowledge_graph, entities, hops=2)

    if not facts:
        return ("No relevant connections found in the knowledge graph "
                "for this question. Try the vector search mode instead.")

    prompt = GRAPH_ANSWER_PROMPT.format(facts=facts, question=question)
    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------
# STEP 7 (NEW): A simple router between vector RAG and graph RAG
# ---------------------------------------------------------------------

ROUTER_PROMPT = """Classify this question into exactly one category:

- "graph" -- if it asks about relationships, connections, or requires
  chaining multiple facts together (e.g. "how is X related to Y",
  "what did the company that acquired X release")
- "vector" -- if it's a direct factual lookup answerable from one
  passage (e.g. "what is X", "when did Y happen")

Return ONLY the single word: graph OR vector

Question: {question}
"""


def route_query(question: str) -> str:
    prompt = ROUTER_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    return "graph" if "graph" in decision else "vector"


# ---------------------------------------------------------------------
# STEP 8: Interactive loop -- routes each question automatically
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nKnowledge graph ready: {knowledge_graph.number_of_nodes()} entities, "
          f"{knowledge_graph.number_of_edges()} relationships.\n")

    while True:
        query = input("\nAsk a question (or 'quit'): ")
        if query.lower() == "quit":
            break

        mode = route_query(query)
        print(f"[router selected: {mode} RAG]")

        if mode == "graph":
            answer = graph_qa(query)
            print("\n--- Answer (Graph RAG) ---")
            print(answer)
        else:
            result = vector_qa_chain.invoke({"query": query})
            print("\n--- Answer (Vector RAG) ---")
            print(result["result"])
            print("\n--- Sources ---")
            for doc in result["source_documents"]:
                print(f"- {doc.metadata.get('source', 'unknown')}")