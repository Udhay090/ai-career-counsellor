"""Gradio UI for the AI Career Counselor."""

import os

import gradio as gr
from crewai import LLM
from dotenv import load_dotenv

from crew import build_crew

load_dotenv()

# CrewAI currently adds cache_breakpoint metadata that some OpenAI-compatible
# providers reject. This patch is harmless if that internal module changes.
try:
    import crewai.llms.cache as _crewai_cache

    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except ImportError:
    pass

MAX_TURNS = 6
MODEL = os.getenv("MODEL", "groq/openai/gpt-oss-120b")

llm_kwargs = {"model": MODEL}
api_key = os.getenv("LLM_API_KEY") or os.getenv(f"{MODEL.split('/', 1)[0].upper()}_API_KEY")
if api_key:
    llm_kwargs["api_key"] = api_key
if os.getenv("LLM_BASE_URL"):
    llm_kwargs.update(base_url=os.environ["LLM_BASE_URL"], custom_openai=True)

llm = LLM(**llm_kwargs)


def format_history(history: list[tuple[str, str]]) -> str:
    recent = history[-MAX_TURNS:]
    return "\n".join(f"User: {u}\nAdvisor: {a}" for u, a in recent) or "(no prior messages)"


def chat(message, chat_display, history_state):
    history_state = history_state or []
    chat_display = chat_display or []
    result = build_crew(llm).kickoff(
        inputs={"query": message, "history": format_history(history_state)}
    )
    answer = str(result)
    history_state = history_state + [(message, answer)]
    chat_display = chat_display + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", chat_display, history_state


def clear_session():
    return [], []


LOAD_JS = """
() => {
    const raw = localStorage.getItem('career_counselor_history');
    const history = raw ? JSON.parse(raw) : [];
    const display = [];
    for (const [user, answer] of history) {
        display.push({role: "user", content: user});
        display.push({role: "assistant", content: answer});
    }
    return [display, history];
}
"""

SAVE_JS = """
(history) => {
    localStorage.setItem('career_counselor_history', JSON.stringify(history));
    return [];
}
"""

CLEAR_JS = """
() => {
    localStorage.removeItem('career_counselor_history');
    return [];
}
"""

with gr.Blocks(title="AI Career Counselor") as demo:
    gr.Markdown("# AI Career Counselor\n4-agent CrewAI pipeline: Profile → Skill Gaps → Courses → Advice")
    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="e.g. Suggest a career in AI", label="Your message")
    clear_btn = gr.Button("Clear session")
    history_state = gr.State([])

    demo.load(None, None, [chatbot, history_state], js=LOAD_JS)
    msg.submit(chat, [msg, chatbot, history_state], [msg, chatbot, history_state]).then(
        None, [history_state], [], js=SAVE_JS
    )
    clear_btn.click(clear_session, None, [chatbot, history_state]).then(
        None, None, [], js=CLEAR_JS
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
