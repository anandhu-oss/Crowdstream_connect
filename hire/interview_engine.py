"""Hire interview engine - V2 (text only, rule-based control + LLM language).

Design
------
* Rules in plain Python decide WHAT happens: which skill is asked next, whether to
  follow up, difficulty, question budget, time limit, coverage of mandatory skills,
  when to stop, no repeated questions.
* The LLM (Qwen3.5-9B behind a vLLM/OpenAI-compatible server) only does language work,
  always with constrained JSON output:  analyze JD+resume, write ONE question, judge ONE answer.
* If the LLM fails, the interview carries on with fallback templates, it never crashes.

Usage
-----
    engine = InterviewEngine(InterviewRules.from_dict(rules_json), jd_text, resume_text,
                             OpenAICompatLLM())
    turn = engine.start()                 # Turn(action="ask", text="...")  -> speak it (V1 TTS)
    while turn.action != "end":
        turn = engine.answer(candidate_text)   # ASR text; "" means silence / unclear audio
    report_input = engine.summary()       # evidence per skill, for the V3 evaluation engine

Quick terminal test (needs an LLM server, see env vars below):

    python -m hire.interview_engine --jd jd.txt --resume resume.txt [--rules rules.json]

Env: HIRE_LLM_URL (default http://localhost:8000/v1), HIRE_LLM_MODEL (default Qwen/Qwen3.5-9B),
     HIRE_LLM_API_KEY (default "EMPTY").
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Protocol

log = logging.getLogger("hire.interview")

DIFFICULTIES = ["beginner", "intermediate", "advanced"]
LEVELS = ["strong", "good", "partial", "incorrect", "insufficient_evidence"]
KIND_ORDER = ["conceptual", "resume", "practical", "scenario-based"]
MAX_QUESTION_WORDS = 70  # questions are spoken aloud, keep them short
DUPLICATE_RATIO = 0.85

CLOSING_TEXT = ("That brings us to the end of the interview. Thank you for your time and "
                "your answers. The hiring team will review the results and get back to you.")
REPEAT_TEXT = "Sorry, I didn't catch that. Let me repeat. "


# --------------------------------------------------------------------------- rules
@dataclass
class InterviewRules:
    """Company-defined rules. Build with InterviewRules.from_dict(json_from_hire_app)."""
    mandatory_skills: list[str] = field(default_factory=list)  # empty -> taken from JD analysis
    duration_minutes: int = 30
    min_questions: int = 10
    max_questions: int = 15
    difficulty_range: tuple[str, str] = ("beginner", "intermediate")
    interview_type: str = "technical"
    question_styles: list[str] = field(
        default_factory=lambda: ["conceptual", "practical", "scenario-based"])
    resume_questions: bool = True
    dynamic_questions: bool = True       # follow-up questions allowed
    max_followups_per_skill: int = 2
    avg_question_seconds: int = 120      # time estimate until real timings are observed

    def __post_init__(self) -> None:
        lo, hi = self.difficulty_range
        if lo not in DIFFICULTIES or hi not in DIFFICULTIES or DIFFICULTIES.index(lo) > DIFFICULTIES.index(hi):
            raise ValueError(f"invalid difficulty_range: {self.difficulty_range}")
        if self.min_questions > self.max_questions:
            raise ValueError("min_questions cannot be greater than max_questions")
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewRules":
        d = dict(d)
        rng = d.pop("difficulty_range", None)
        text = d.pop("difficulty", None)
        if rng is None and text:
            found = [w for w in DIFFICULTIES if w in str(text).lower()]
            if found:
                rng = (found[0], found[-1])
        if rng is not None:
            d["difficulty_range"] = (rng[0], rng[-1])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- LLM client
class LLM(Protocol):
    def json(self, task: str, system: str, user: str, schema: dict) -> dict: ...


class OpenAICompatLLM:
    """Talks to a vLLM / SGLang OpenAI-compatible server with JSON-schema constrained output."""
    TEMPERATURE = {"analyze": 0.0, "judge": 0.0, "question": 0.7}

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or os.getenv("HIRE_LLM_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("HIRE_LLM_MODEL", "Qwen/Qwen3.5-9B")
        self.api_key = api_key or os.getenv("HIRE_LLM_API_KEY", "EMPTY")
        self.timeout = timeout

    def json(self, task: str, system: str, user: str, schema: dict) -> dict:
        import httpx

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.TEMPERATURE.get(task, 0.3),
            "max_tokens": 700,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": task, "schema": schema, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False},  # no thinking in live turns
        }
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                       headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


# --------------------------------------------------------------------------- schemas & prompts
ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "candidate_skills": {"type": "array", "items": {"type": "string"}},
        "years_experience": {"type": "number"},
        "projects": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "summary": {"type": "string"},
                           "skills": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "summary", "skills"]}},
    },
    "required": ["role", "responsibilities", "required_skills", "candidate_skills",
                 "years_experience", "projects"],
}
QUESTION_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}},
                   "required": ["question"]}
JUDGE_SCHEMA = {"type": "object",
                "properties": {"level": {"type": "string", "enum": LEVELS},
                               "evidence": {"type": "string"}},
                "required": ["level", "evidence"]}

ANALYZE_SYSTEM = (
    "You extract structured data from a job description and a candidate resume. "
    "Return only facts that are written in the texts. required_skills come from the job "
    "description, candidate_skills and projects from the resume. Keep each list short and specific.")

QUESTION_SYSTEM = (
    "You are a professional interviewer running a spoken {type} interview for the {role} role. "
    "Write exactly ONE interview question. It will be read aloud by a text-to-speech voice, so "
    "use plain spoken English: one or two short sentences, no lists, no markdown, no code, no "
    "symbols that are hard to say aloud. Do not give hints or the answer. Never repeat or "
    "rephrase an earlier question.")

JUDGE_SYSTEM = (
    "You are an objective technical interview assessor. Judge ONLY the technical content of the "
    "answer. The answer was transcribed automatically from speech, so ignore grammar, accent, "
    "filler words and likely transcription errors, and do not judge communication style.\n"
    "Levels: strong (accurate, deep, with examples); good (accurate and adequate); partial "
    "(some correct points but incomplete); incorrect (clearly wrong claims or misunderstanding); "
    "insufficient_evidence (too vague, too short, off-topic or 'I don't know', so we cannot tell "
    "what the candidate knows). Do not mark vague answers as incorrect.\n"
    "evidence: one sentence naming what the candidate actually said that justifies the level.")

KIND_HELP = {
    "conceptual": "Test understanding of a core concept of the skill.",
    "practical": "Ask how they would do, or have done, something hands-on with the skill.",
    "scenario-based": "Describe a short realistic work scenario and ask how they would handle it.",
    "resume": "Ask about the candidate's own project below, with concrete implementation detail.",
    "followup": "Ask ONE short follow-up that probes the gap in the previous answer.",
}


# --------------------------------------------------------------------------- state
@dataclass
class QA:
    index: int
    skill: Optional[str]
    kind: str
    difficulty: str
    question: str
    followup: bool = False
    answer: str = ""
    level: str = ""
    evidence: str = ""
    seconds: float = 0.0
    unjudged: bool = False   # silence or judge failure: no follow-up, no evidence


@dataclass
class Turn:
    action: str              # "ask" | "repeat" | "end"
    text: str
    skill: Optional[str] = None


@dataclass
class Plan:
    skill: str
    kind: str
    difficulty: str
    followup_of: Optional[QA] = None
    project: Optional[dict] = None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def _is_duplicate(question: str, previous: list[str]) -> bool:
    nq = _norm(question)
    return any(difflib.SequenceMatcher(None, nq, _norm(p)).ratio() >= DUPLICATE_RATIO
               for p in previous)


# --------------------------------------------------------------------------- engine
class InterviewEngine:
    def __init__(self, rules: InterviewRules, jd: str, resume: str, llm: LLM,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.rules = rules
        self.jd = jd
        self.resume = resume
        self.llm = llm
        self._clock = clock

        self.profile: dict = {}
        self.skills: list[str] = list(rules.mandatory_skills)
        self.history: list[QA] = []
        self.candidate_intro = ""
        self.ended_reason: Optional[str] = None

        self._t0 = 0.0
        self._asked_at = 0.0
        self._started = False
        self._ended = False
        self._intro_pending = False
        self._current: Optional[QA] = None
        self._current_text = ""
        self._repeated = False
        self._skill_level_idx: dict[str, int] = {}
        self._used_projects: set[str] = set()

    # ---- public API ----
    def start(self) -> Turn:
        self._t0 = self._clock()
        self._started = True
        self.profile = self._analyze()
        if not self.skills:
            self.skills = list(self.profile.get("required_skills", []))[:5]
        if not self.skills:
            raise ValueError("No mandatory skills: provide them in the rules or in the JD")
        lo = DIFFICULTIES.index(self.rules.difficulty_range[0])
        self._skill_level_idx = {s: lo for s in self.skills}

        role = self.profile.get("role") or "open"
        text = (f"Hello, I'm your AI interviewer for the {role} position. To start, could you "
                "briefly introduce yourself and tell me about your background?")
        self._intro_pending = True
        return self._ask(text, None)

    def answer(self, text: str) -> Turn:
        if not self._started:
            raise RuntimeError("call start() first")
        if self._ended:
            return Turn("end", CLOSING_TEXT)

        text = (text or "").strip()
        if not text:                                   # silence / unclear audio
            if not self._repeated:
                self._repeated = True
                return Turn("repeat", REPEAT_TEXT + self._current_text,
                            self._current.skill if self._current else None)
            self._repeated = False                     # second silence: record "no answer", move on
            return self._finish_answer("", unjudged=True)
        self._repeated = False
        return self._finish_answer(text)

    def summary(self) -> dict:
        skills = {}
        for s in self.skills:
            qas = [q for q in self.history if q.skill == s]
            skills[s] = {
                "questions": sum(1 for q in qas if not q.followup),
                "followups": sum(1 for q in qas if q.followup),
                "levels": [q.level for q in qas],
                "evidence": [q.evidence for q in qas if q.evidence],
            }
        return {
            "role": self.profile.get("role", ""),
            "elapsed_seconds": round(self._clock() - self._t0, 1) if self._started else 0,
            "ended_reason": self.ended_reason,
            "uncovered_skills": [s for s in self.skills if skills[s]["questions"] == 0],
            "candidate_intro": self.candidate_intro,
            "skills": skills,
            "transcript": [asdict(q) for q in self.history],
        }

    # ---- turn handling ----
    def _ask(self, text: str, skill: Optional[str]) -> Turn:
        self._current_text = text
        self._asked_at = self._clock()
        return Turn("ask", text, skill)

    def _finish_answer(self, text: str, unjudged: bool = False) -> Turn:
        now = self._clock()
        if self._intro_pending:
            self.candidate_intro = text[:800]
            self._intro_pending = False
        elif self._current is not None:
            qa = self._current
            qa.answer = text
            qa.seconds = now - self._asked_at
            if unjudged:
                qa.level, qa.evidence, qa.unjudged = "insufficient_evidence", "No answer given.", True
            else:
                self._judge(qa)
        return self._next_turn()

    def _next_turn(self) -> Turn:
        reason = self._end_reason()
        if reason:
            self._ended, self.ended_reason, self._current = True, reason, None
            return Turn("end", CLOSING_TEXT)

        plan = self._decide_next()
        question = self._generate_question(plan)
        if plan.project:
            self._used_projects.add(plan.project["name"])
        qa = QA(index=len(self.history) + 1, skill=plan.skill, kind=plan.kind,
                difficulty=plan.difficulty, question=question,
                followup=plan.followup_of is not None)
        self.history.append(qa)
        self._current = qa
        return self._ask(question, plan.skill)

    # ---- rules ----
    def _elapsed(self) -> float:
        return self._clock() - self._t0

    def _primary_count(self, skill: str) -> int:
        return sum(1 for q in self.history if q.skill == skill and not q.followup)

    def _end_reason(self) -> Optional[str]:
        r = self.rules
        if self._elapsed() >= r.duration_minutes * 60:
            return "time_limit"
        if len(self.history) >= r.max_questions:
            return "max_questions"
        if len(self.history) >= r.min_questions and all(self._primary_count(s) for s in self.skills):
            return "completed"
        return None

    def _avg_seconds(self) -> float:
        done = [q.seconds for q in self.history if q.seconds > 0]
        return sum(done) / len(done) if done else float(self.rules.avg_question_seconds)

    def _decide_next(self) -> Plan:
        r = self.rules
        last = self.history[-1] if self.history else None
        uncovered = [s for s in self.skills if self._primary_count(s) == 0]
        remaining_q = r.max_questions - len(self.history)
        time_left = r.duration_minutes * 60 - self._elapsed()

        # 1) Coverage first: if the budget is tight, only uncovered mandatory skills are asked.
        tight = bool(uncovered) and (len(uncovered) >= remaining_q or
                                     time_left <= self._avg_seconds() * len(uncovered) * 1.2)
        if tight:
            return self._plan_primary(uncovered[0])

        # 2) Follow up on a partial / vague answer (bounded).
        if (r.dynamic_questions and last and last.skill and not last.unjudged
                and last.level in ("partial", "insufficient_evidence")
                and sum(1 for q in self.history if q.skill == last.skill and q.followup)
                < r.max_followups_per_skill):
            return Plan(last.skill, "followup", last.difficulty, followup_of=last)

        # 3) Otherwise the least-covered mandatory skill (ties: order given by the company).
        skill = min(self.skills, key=lambda s: (self._primary_count(s), self.skills.index(s)))
        return self._plan_primary(skill)

    def _plan_primary(self, skill: str) -> Plan:
        kind, project = self._pick_kind(skill)
        difficulty = DIFFICULTIES[self._skill_level_idx[skill]]
        return Plan(skill, kind, difficulty, project=project)

    def _project_for(self, skill: str) -> Optional[dict]:
        if not self.rules.resume_questions:
            return None
        for p in self.profile.get("projects", []):
            blob = " ".join([p.get("name", ""), p.get("summary", ""), *p.get("skills", [])]).lower()
            if skill.lower() in blob and p.get("name") not in self._used_projects:
                return p
        return None

    def _pick_kind(self, skill: str) -> tuple[str, Optional[dict]]:
        n = self._primary_count(skill)
        allowed = set(self.rules.question_styles)
        for i in range(n, n + len(KIND_ORDER)):
            kind = KIND_ORDER[i % len(KIND_ORDER)]
            if kind == "resume":
                project = self._project_for(skill)
                if project:
                    return "resume", project
            elif kind in allowed:
                return kind, None
        return (self.rules.question_styles or ["conceptual"])[0], None

    def _adapt_difficulty(self, skill: str, level: str) -> None:
        lo = DIFFICULTIES.index(self.rules.difficulty_range[0])
        hi = DIFFICULTIES.index(self.rules.difficulty_range[1])
        idx = self._skill_level_idx[skill]
        if level == "strong":
            idx += 1
        elif level == "incorrect":
            idx -= 1
        self._skill_level_idx[skill] = max(lo, min(hi, idx))

    # ---- LLM steps (each one falls back instead of failing) ----
    def _analyze(self) -> dict:
        user = f"JOB DESCRIPTION:\n{self.jd[:12000]}\n\nRESUME:\n{self.resume[:12000]}"
        try:
            return self.llm.json("analyze", ANALYZE_SYSTEM, user, ANALYZE_SCHEMA)
        except Exception:
            log.exception("JD/resume analysis failed; continuing with rule skills only")
            return {"role": "", "responsibilities": [], "required_skills": list(self.skills),
                    "candidate_skills": [], "projects": [], "years_experience": 0}

    def _generate_question(self, plan: Plan) -> str:
        role = self.profile.get("role") or "open"
        system = QUESTION_SYSTEM.format(type=self.rules.interview_type, role=role)
        previous = [q.question for q in self.history]
        user = self._question_prompt(plan, previous)
        for attempt in range(3):
            try:
                q = str(self.llm.json("question", system, user, QUESTION_SCHEMA).get("question", "")).strip()
            except Exception:
                log.warning("question generation failed (attempt %d)", attempt + 1, exc_info=True)
                continue
            if not q or len(q.split()) > MAX_QUESTION_WORDS:
                continue
            if _is_duplicate(q, previous):
                log.info("duplicate question rejected, regenerating")
                continue
            return q
        return self._fallback_question(plan)

    def _question_prompt(self, plan: Plan, previous: list[str]) -> str:
        lines = [f"Skill to assess: {plan.skill}",
                 f"Difficulty: {plan.difficulty}",
                 f"Question type: {plan.kind}. {KIND_HELP[plan.kind]}"]
        resp = self.profile.get("responsibilities", [])[:5]
        if resp:
            lines.append("Role responsibilities: " + "; ".join(resp))
        if self.candidate_intro:
            lines.append(f"Candidate introduction (context only): {self.candidate_intro[:400]}")
        if plan.project:
            lines.append(f"Candidate project: {plan.project.get('name')} - {plan.project.get('summary')}")
        if plan.followup_of:
            f = plan.followup_of
            lines += [f"Previous question: {f.question}",
                      f"Candidate answer: {f.answer[:600]}",
                      f"Gap to probe: {f.evidence}"]
        if previous:
            lines.append("Questions already asked (do not repeat):\n- " + "\n- ".join(previous))
        return "\n".join(lines)

    @staticmethod
    def _fallback_question(plan: Plan) -> str:
        s = plan.skill
        return {
            "followup": f"Could you go into a bit more detail on your last answer about {s}?",
            "resume": f"Tell me about a project where you used {s} and what your part in it was.",
            "practical": f"How have you used {s} in practice? Walk me through a concrete example.",
            "scenario-based": f"Imagine a problem comes up at work that involves {s}. How would you approach it?",
        }.get(plan.kind, f"Can you explain the key concepts of {s} in your own words?")

    def _judge(self, qa: QA) -> None:
        user = (f"Skill: {qa.skill}\nDifficulty: {qa.difficulty}\n"
                f"Question: {qa.question}\nCandidate answer (auto-transcribed): {qa.answer}")
        try:
            res = self.llm.json("judge", JUDGE_SYSTEM, user, JUDGE_SCHEMA)
            level = res.get("level")
            if level not in LEVELS:
                raise ValueError(f"bad level {level!r}")
            qa.level, qa.evidence = level, str(res.get("evidence", "")).strip()
        except Exception:
            log.exception("answer judging failed for question %d", qa.index)
            qa.level, qa.evidence, qa.unjudged = "insufficient_evidence", "Judge unavailable.", True
            return
        if qa.skill:
            self._adapt_difficulty(qa.skill, qa.level)


# --------------------------------------------------------------------------- terminal test
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Text-only interview test")
    ap.add_argument("--jd", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--rules", help="JSON file with interview rules")
    ap.add_argument("--url")
    ap.add_argument("--model")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rules = (InterviewRules.from_dict(json.load(open(args.rules))) if args.rules
             else InterviewRules())
    engine = InterviewEngine(rules, open(args.jd).read(), open(args.resume).read(),
                             OpenAICompatLLM(args.url, args.model))
    turn = engine.start()
    try:
        while True:
            print(f"\nAI: {turn.text}")
            if turn.action == "end":
                break
            turn = engine.answer(input("You (empty = silence): "))
    except (EOFError, KeyboardInterrupt):
        pass
    print("\n--- summary ---")
    print(json.dumps(engine.summary(), indent=2))


if __name__ == "__main__":
    main()
