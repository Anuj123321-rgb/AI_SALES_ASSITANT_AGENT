from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


Intent = Literal["greeting", "product_inquiry", "high_intent_lead"]


class AgentState(TypedDict):
    messages: List[BaseMessage]
    intent: Optional[Intent]
    lead: Dict[str, Optional[str]]
    lead_captured: bool


@dataclass
class AutoStreamConfig:
    kb_path: str = "knowledge_base.md"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2


def mock_lead_capture(name: str, email: str, platform: str) -> None:
    print(f"Lead captured successfully: {name}, {email}, {platform}")


class AutoStreamAgent:
    def __init__(self, config: AutoStreamConfig | None = None):
        self.config = config or AutoStreamConfig()

        with open(self.config.kb_path, "r", encoding="utf-8") as f:
            kb_text = f.read()

        self.retriever = BM25Retriever.from_texts([kb_text])
        self.retriever.k = 1

        self.llm = ChatOpenAI(model=self.config.model, temperature=self.config.temperature)
        self.intent_prompt = ChatPromptTemplate.from_template(
            """
You are an intent classifier for AutoStream.
Classify the user message into exactly one label:
- greeting
- product_inquiry
- high_intent_lead

User message: {message}

Respond with label only.
""".strip()
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            """
You are AutoStream's sales assistant.
Answer using ONLY the provided context. If missing, say you only know what is in policy/pricing docs.
Keep answer concise and helpful.

Context:
{context}

User question:
{question}
""".strip()
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("assistant_step", self._assistant_step)
        builder.add_edge(START, "assistant_step")
        builder.add_edge("assistant_step", END)
        return builder.compile()

    def _classify_intent(self, message: str) -> Intent:
        quick = message.lower()
        high_intent_keywords = [
            "sign up",
            "start",
            "subscribe",
            "buy",
            "purchase",
            "try pro",
            "want pro",
            "get pro",
        ]
        if any(k in quick for k in high_intent_keywords):
            return "high_intent_lead"

        chain = self.intent_prompt | self.llm
        raw = chain.invoke({"message": message}).content.strip().lower()
        if "high_intent_lead" in raw:
            return "high_intent_lead"
        if "product_inquiry" in raw:
            return "product_inquiry"
        return "greeting"

    @staticmethod
    def _extract_email(text: str) -> Optional[str]:
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_platform(text: str) -> Optional[str]:
        known = ["youtube", "instagram", "tiktok", "linkedin", "x", "facebook", "twitch"]
        lowered = text.lower()
        for item in known:
            if item in lowered:
                return item.title()
        return None

    @staticmethod
    def _extract_name(text: str) -> Optional[str]:
        patterns = [
            r"(?:i am|i'm|my name is)\s+([A-Za-z][A-Za-z\s'-]{1,40})",
            r"name[:\s]+([A-Za-z][A-Za-z\s'-]{1,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        return None

    def _update_lead_fields(self, text: str, lead: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        lead = dict(lead)
        lead["email"] = lead.get("email") or self._extract_email(text)
        lead["platform"] = lead.get("platform") or self._extract_platform(text)
        lead["name"] = lead.get("name") or self._extract_name(text)
        return lead

    @staticmethod
    def _missing_fields(lead: Dict[str, Optional[str]]) -> List[str]:
        order = ["name", "email", "platform"]
        return [field for field in order if not lead.get(field)]

    def _ask_for_missing_details(self, lead: Dict[str, Optional[str]]) -> str:
        missing = self._missing_fields(lead)
        labels = {
            "name": "your name",
            "email": "your email",
            "platform": "your creator platform (YouTube, Instagram, etc.)",
        }
        if not missing:
            return "Thanks — I have all details required for signup."
        if len(missing) == 1:
            return f"Great! To proceed, please share {labels[missing[0]]}."
        pending = ", ".join(labels[item] for item in missing[:-1]) + f" and {labels[missing[-1]]}"
        return f"Awesome — to continue with signup, please share {pending}."

    def _rag_answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        chain = self.answer_prompt | self.llm
        return chain.invoke({"context": context, "question": question}).content

    def _assistant_step(self, state: AgentState) -> AgentState:
        user_msg = state["messages"][-1].content
        intent = self._classify_intent(user_msg)
        lead = self._update_lead_fields(user_msg, state["lead"])
        lead_captured = state["lead_captured"]

        if intent == "high_intent_lead" or any(lead.values()):
            missing = self._missing_fields(lead)
            if not missing and not lead_captured:
                mock_lead_capture(
                    name=lead["name"] or "",
                    email=lead["email"] or "",
                    platform=lead["platform"] or "",
                )
                lead_captured = True
                reply = (
                    "Perfect, you're all set! I've captured your lead for the Pro plan. "
                    "Our team will reach out shortly."
                )
            elif not missing and lead_captured:
                reply = "Your lead is already captured. Would you like plan comparison details as well?"
            else:
                reply = self._ask_for_missing_details(lead)
        elif intent == "product_inquiry":
            reply = self._rag_answer(user_msg)
        else:
            reply = (
                "Hi! I'm AutoStream's assistant. I can help with pricing, features, and signup. "
                "What would you like to know?"
            )

        new_messages = state["messages"] + [AIMessage(content=reply)]
        return {
            "messages": new_messages,
            "intent": intent,
            "lead": lead,
            "lead_captured": lead_captured,
        }

    def chat(self, text: str, state: Optional[AgentState] = None) -> AgentState:
        if state is None:
            state = {
                "messages": [],
                "intent": None,
                "lead": {"name": None, "email": None, "platform": None},
                "lead_captured": False,
            }
        state["messages"] = state["messages"] + [HumanMessage(content=text)]
        return self.graph.invoke(state)
