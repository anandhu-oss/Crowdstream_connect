"""Tests for the V2 interview engine. No GPU or LLM server needed (scripted fake LLM)."""
import random

import pytest

try:
    from hire import interview_engine as ie
except ImportError:  # flat layout
    import interview_engine as ie

VOCAB = ("river table orange engine violin window carpet planet bridge candle garden hammer "
         "pencil ladder mirror pillow saddle ticket anchor basket cloud dragon feather glacier "
         "harbor island jungle kettle lantern meadow needle oyster pepper quartz ribbon sunset "
         "tunnel umbrella valley whistle yogurt zipper").split()

PROFILE = {
    "role": "Junior Python Developer",
    "responsibilities": ["Build REST APIs", "Write SQL queries"],
    "required_skills": ["Python", "SQL", "FastAPI"],
    "candidate_skills": ["Python", "FastAPI"],
    "years_experience": 1,
    "projects": [{"name": "Shop API", "summary": "REST API built with FastAPI and SQL",
                  "skills": ["FastAPI", "SQL"]}],
}


class FakeLLM:
    def __init__(self, levels=None, scripted_questions=None):
        self.levels = list(levels or [])
        self.scripted = list(scripted_questions or [])
        self.counts = {"analyze": 0, "question": 0, "judge": 0}
        self.n = 0

    def json(self, task, system, user, schema):
        self.counts[task] += 1
        if task == "analyze":
            return PROFILE
        if task == "judge":
            level = self.levels.pop(0) if self.levels else "good"
            return {"level": level, "evidence": f"said something rated {level}"}
        if self.scripted:
            return {"question": self.scripted.pop(0)}
        self.n += 1
        words = random.Random(self.n).choices(VOCAB, k=10)
        return {"question": " ".join(words) + "?"}


class BrokenLLM:
    def json(self, *a, **k):
        raise RuntimeError("server down")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(rules=None, llm=None, clock=None, **rule_kw):
    rules = rules or ie.InterviewRules(mandatory_skills=["Python", "SQL", "FastAPI"], **rule_kw)
    return ie.InterviewEngine(rules, "jd", "resume", llm or FakeLLM(), clock or Clock())


def run(engine, answer="Here is my detailed answer about it.", limit=100):
    turn = engine.start()
    assert turn.action == "ask"
    for _ in range(limit):
        turn = engine.answer(answer)
        if turn.action == "end":
            return
    pytest.fail("interview never ended")


def test_covers_all_mandatory_skills_and_stops_when_done():
    eng = make(min_questions=4, max_questions=10)
    run(eng)
    s = eng.summary()
    assert s["uncovered_skills"] == []
    assert 4 <= len(eng.history) <= 10
    assert s["ended_reason"] == "completed"


def test_partial_answers_trigger_bounded_followups():
    eng = make(rules=ie.InterviewRules(mandatory_skills=["Python", "SQL"], min_questions=5,
                                       max_questions=6, max_followups_per_skill=2),
               llm=FakeLLM(levels=["partial"] * 20))
    run(eng)
    h = eng.history
    assert [(q.skill, q.followup) for q in h[:4]] == [
        ("Python", False), ("Python", True), ("Python", True), ("SQL", False)]


def test_incorrect_answer_moves_on():
    eng = make(rules=ie.InterviewRules(mandatory_skills=["Python", "SQL"], min_questions=4,
                                       max_questions=6),
               llm=FakeLLM(levels=["incorrect"]))
    run(eng)
    assert eng.history[1].followup is False
    assert eng.history[1].skill == "SQL"


def test_tight_budget_forces_coverage_without_followups():
    eng = make(min_questions=3, max_questions=3, llm=FakeLLM(levels=["partial"] * 10))
    run(eng)
    assert [q.skill for q in eng.history] == ["Python", "SQL", "FastAPI"]
    assert not any(q.followup for q in eng.history)


def test_dynamic_questions_disabled_means_no_followups():
    eng = make(min_questions=4, max_questions=6, dynamic_questions=False,
               llm=FakeLLM(levels=["partial"] * 10))
    run(eng)
    assert not any(q.followup for q in eng.history)


def test_duplicate_question_is_regenerated():
    llm = FakeLLM(scripted_questions=[
        "Explain how Python manages memory for objects.",
        "Explain how Python manages memory for objects.",   # duplicate -> must be rejected
        "What does a database index actually speed up?",
    ])
    eng = make(min_questions=2, max_questions=2, llm=llm)
    eng.start()
    eng.answer("I am a developer with one year of experience.")   # intro -> question 1
    eng.answer("Reference counting and a garbage collector.")      # -> question 2
    qs = [q.question for q in eng.history]
    assert qs[0] != qs[1]
    assert llm.counts["question"] == 3


def test_time_limit_is_a_hard_stop():
    clock = Clock()
    eng = make(clock=clock, duration_minutes=30)
    eng.start()
    clock.t = 31 * 60
    turn = eng.answer("My introduction goes here.")
    assert turn.action == "end"
    assert eng.summary()["ended_reason"] == "time_limit"
    assert eng.summary()["uncovered_skills"] == ["Python", "SQL", "FastAPI"]


def test_silence_repeats_once_then_moves_on_without_followup():
    eng = make(min_questions=3, max_questions=5)
    eng.start()
    assert eng.answer("").action == "repeat"            # intro: repeat
    t = eng.answer("")                                   # still silent: move on
    assert t.action == "ask" and len(eng.history) == 1
    repeat = eng.answer("   ")                           # question 1: repeat once
    assert repeat.action == "repeat" and eng.history[0].question in repeat.text
    assert eng.answer("").action == "ask"                # second silence -> next question
    first = eng.history[0]
    assert first.level == "insufficient_evidence" and first.unjudged
    assert eng.history[1].followup is False


def test_llm_failure_never_crashes_the_interview():
    eng = make(llm=BrokenLLM(), min_questions=3, max_questions=4)
    run(eng)
    assert all(q.question for q in eng.history)
    assert all(q.unjudged for q in eng.history)
    assert not any(q.followup for q in eng.history)


def test_difficulty_adapts_within_company_bounds():
    eng = make(rules=ie.InterviewRules(mandatory_skills=["Python"], min_questions=4,
                                       max_questions=4,
                                       difficulty_range=("beginner", "intermediate")),
               llm=FakeLLM(levels=["strong"] * 10))
    run(eng)
    d = [q.difficulty for q in eng.history]
    assert d[0] == "beginner" and d[-1] == "intermediate"
    assert set(d) <= {"beginner", "intermediate"}


def test_resume_question_uses_candidate_project():
    eng = make(min_questions=6, max_questions=8)
    run(eng)
    assert any(q.kind == "resume" for q in eng.history)


def test_skills_default_to_jd_when_rules_have_none():
    eng = ie.InterviewEngine(ie.InterviewRules(min_questions=3, max_questions=4), "jd", "cv",
                             FakeLLM(), Clock())
    run(eng)
    assert eng.skills == ["Python", "SQL", "FastAPI"]


def test_rules_from_dict_and_validation():
    r = ie.InterviewRules.from_dict({"difficulty": "Beginner to intermediate",
                                     "mandatory_skills": ["Python"], "unknown_key": 1})
    assert r.difficulty_range == ("beginner", "intermediate")
    with pytest.raises(ValueError):
        ie.InterviewRules(min_questions=10, max_questions=5)
    with pytest.raises(ValueError):
        ie.InterviewRules(difficulty_range=("advanced", "beginner"))
