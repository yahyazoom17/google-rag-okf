import os
import glob
import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from laya import Router

load_dotenv()

laya_router = Router()

KNOWLEDGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company-knowledge")
SIMPLE_MODEL = "openrouter/nvidia/nemotron-3.5-super-vl:free"   
SUPER_MODEL = "openai/gpt-5.4" 
MIN_CONFIDENCE = 0.7           

docs = {}

for path in glob.glob(KNOWLEDGE_FOLDER + "/**/*.md", recursive=True):
    text = open(path, encoding="utf-8").read()
    frontmatter = text.split("---")[1]
    meta = yaml.safe_load(frontmatter)

    key = os.path.basename(path).replace(".md", "").replace("-", "_")
    docs[key] = {"path": path, "meta": meta, "text": text}

print("Loaded", len(docs), "OKF files:", list(docs.keys()))

while True:
    question = input("\nAsk a question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        break
    if question == "":
        continue

    file_options = {}
    for key in docs:
        file_options[key] = docs[key]["meta"]["description"]
    file_options["none"] = "No knowledge file is relevant to this question"

    result = laya_router.predict(
        {"question": question},
        {
            "doc": {
                "type": "choice",
                "instructions": (
                    "Which company knowledge file best answers "
                    "the user's question? Choose none if no file is relevant."
                ),
                "criteria": file_options,
            },
            "level": {
                "type": "choice",
                "instructions": (
                    "Classify the complexity of the user's question."
                ),
                "criteria": {
                    "simple": "A short fact, definition or quick lookup",
                    "super": (
                        "Requires analysis, planning, comparison, "
                        "code or multiple steps"
                    ),
                },
            },
        },
    )

    doc_answer = result["answers"]["doc"]
    level_answer = result["answers"]["level"]

    print("\n[Laya] Confidence scores:")    
    print("  - File:", round(doc_answer["confidence"], 2))
    print("  - Level:", round(level_answer["confidence"], 2))

    picked_file = doc_answer["choice"]
    picked_level = level_answer["choice"]

    print(
        "\n[Laya] File:", picked_file,
        "| confidence:", round(doc_answer["confidence"], 2)
    )
    print(
        "[Laya] Level:", picked_level,
        "| confidence:", round(level_answer["confidence"], 2)
    )

    if picked_level == "super":
        model = SUPER_MODEL
    else:
        model = SIMPLE_MODEL

    llm_config = LLM(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        temperature=0.2,
    )

    print("[Route] LLM:", model)

    if picked_file != "none" and doc_answer["confidence"] >= MIN_CONFIDENCE:
        files_to_read = [picked_file] + docs[picked_file]["meta"].get("links", [])
    else:
        files_to_read = list(docs.keys())

    knowledge = ""
    for key in files_to_read:
        if key in docs:
            knowledge = knowledge + "\n\n===== FILE: " + key + " =====\n" + docs[key]["text"]

    print("[Route] Model:", model)
    print("[Route] Files:", files_to_read)

    agent = Agent(
        role="Company Data Assistant",
        goal="Answer questions using ONLY the company knowledge files you are given",
        backstory="You follow the company's definitions exactly. If the answer is not in the files, you say so and never guess.",
        llm=llm_config,
    )

    task = Task(
        description="Question: " + question + "\n\nCompany knowledge files:" + knowledge,
        expected_output="A clear, correct answer that names which knowledge files it used",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    answer = crew.kickoff()

    print("\n========== ANSWER ==========")
    print(answer.raw)
