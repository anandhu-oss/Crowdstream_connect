"""Prompt templates. Kept separate so they can be tuned without touching logic.

Each user prompt starts with a ``TASK:<KIND>`` marker (used by the mock engine and
handy when reading server logs). Content is fenced with <<<NAME ... NAME>>> so the
model can't confuse transcript text with instructions.

Design notes (what these prompts are tuned for):
  * completeness: every decision / commitment / date must survive, so the system prompt
    says so explicitly and a deterministic date checklist is appended after the transcript;
  * precision: decisions need real agreement, owners/deadlines are never guessed;
  * topic-level key points instead of one bullet per sentence;
  * one short worked example (fictional meeting) to pin down the JSON and the judgement calls.
"""
from __future__ import annotations

from typing import Optional

from .schemas import SummaryPreferences

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ml": "Malayalam", "ta": "Tamil", "de": "German",
    "fr": "French", "es": "Spanish", "it": "Italian", "ar": "Arabic", "ja": "Japanese",
    "zh": "Chinese", "te": "Telugu", "kn": "Kannada", "bn": "Bengali", "mr": "Marathi",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "pt": "Portuguese", "ru": "Russian",
    "ko": "Korean",
}


def resolve_language(value: Optional[str], default: str = "English") -> str:
    """Accept 'ml', 'ml-IN', 'Malayalam' ... and return a language name for the prompt."""
    if not value:
        return default
    v = value.strip()
    code = v.lower().replace("_", "-").split("-")[0]
    if len(v) <= 5 and code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code]
    return v[:1].upper() + v[1:]


SYSTEM_PROMPT = """You are a meticulous meeting-minutes assistant. You read transcripts that may be \
multilingual or code-mixed (for example Malayalam-English, Hindi-English, Tamil-English) and that \
may contain speech-recognition errors.

Rules:
1. Use ONLY information stated in the transcript. Never invent decisions, task owners, deadlines, \
numbers or names.
2. Completeness matters as much as accuracy: do not drop any decision, commitment, date, time, \
number or name that was spoken.
3. Keep names, technical terms, numbers and dates exactly as spoken; do not translate names. Keep \
relative dates ("today", "Friday") as spoken; never convert them to calendar dates.
4. Understand the transcript in its original language(s); do not translate it first. Write the \
output in the requested language.
5. Speaker labels may be generic (e.g. "Speaker 1"). Attribute by label; never guess real names.
6. If a passage is unclear or garbled, say less rather than guess.
7. Missing information is null (or an empty list), never a guess.
8. Respond with ONE valid JSON object and nothing else: no markdown fences, no commentary."""

CHUNK_SHAPE = """{
  "topics": [{"title": "short title", "summary": "1-2 sentences", "timestamp": "HH:MM:SS or null"}],
  "decisions": [{"decision": "clearly agreed decision", "timestamp": "HH:MM:SS or null"}],
  "action_items": [{"task": "...", "assigned_to": null, "deadline": null, "status": "open", "source_timestamp": null}],
  "dates": [{"date": "as spoken, e.g. 'December 5' or 'today'", "what": "what it refers to", "owner": null, "timestamp": null}],
  "unresolved": ["open question, blocker, pending item or deferred topic"],
  "facts": ["important number, name, figure or exact statement worth keeping"]
}"""

FINAL_SHAPE = """{
  "meeting_overview": "1-2 sentences: what kind of meeting, who took part, its purpose",
  "executive_summary": "one paragraph of 3-6 sentences: main discussion, outcomes, key dates, what happens next",
  "key_discussion_points": ["..."],
  "decisions": ["clearly agreed decisions only; [] if none"],
  "action_items": [{"task": "...", "assigned_to": null, "deadline": null, "status": "open|in_progress|done|unspecified", "source_timestamp": null}],
  "key_dates": [{"date": "as spoken", "what": "what it refers to", "owner": null, "timestamp": null}],
  "unresolved_issues": ["..."],
  "next_steps": ["agreed follow-up activities; [] if none"]
}"""

FIELD_GUIDE = """How to fill each field:
- key_discussion_points: topic-level points (merge related remarks). NOT one bullet per sentence. \
Each point names the topic and its substance (who, numbers, dates). Do not just repeat decisions or action items here; cover the status and context behind them.
- decisions: only what participants clearly agreed or confirmed. A plan, date or approach proposed \
by one person and accepted by the others counts, and a closing "agreed"/"okay" from the group \
confirms the proposals made just before it. Ideas, "maybe", questions and plain status reports are \
NOT decisions.
- action_items: one entry per distinct future task that someone committed to, or was asked to do \
and accepted. A task the group agreed to do together with no named owner is still an action item \
(assigned_to null): this includes scheduled group activities such as "We'll conduct final testing on December 2" -> task "Conduct final testing", assigned_to null, deadline "December 2". Not action items: work already finished, plain status updates, vague ideas \
("we could ...", "maybe").
  assigned_to = the person (as named or labelled) who took the task; never guess. \
deadline = the time limit in the speaker's own words; if two limits are given for one task keep \
both (e.g. "contact today; resolve by November 28").
- key_dates: EVERY date, time, deadline, milestone, scheduled event and promised duration that was \
mentioned (project deadlines, releases, reviews, test dates, "two more days", ...), each with what \
it refers to and the owner if explicit. Nothing date-like may be left out.
- unresolved_issues: open questions, decisions postponed, blockers and pending dependencies \
(e.g. "X is still pending"), risks raised without a resolution, and anything lacking an owner or \
an outcome.
- next_steps: the agreed follow-up activities in order, including scheduled follow-up meetings."""

EXAMPLE = """<<<EXAMPLE (a different, fictional meeting - never reuse its names, dates or content)
Dana: Budget approval is due by 15 March.
Lee: The vendor quotes are still missing. I'll chase them tomorrow and send them by Friday.
Dana: Okay. Omar, can you prepare the slides for the 15 March meeting?
Omar: Yes.
Dana: We'll reuse the old template, everyone fine with that?
Lee: Fine.
Omar: Fine.
Dana: Let's do a dry run together on 12 March.
Omar: Sounds good.
Correct JSON:
{"meeting_overview":"Budget planning call between Dana, Lee and Omar ahead of the 15 March approval.","executive_summary":"Dana set 15 March as the budget approval deadline. Lee reported that the vendor quotes are still missing and will chase them; Omar will prepare the slides. The group agreed to reuse the old slide template and to do a dry run together on 12 March.","key_discussion_points":["Budget approval deadline of 15 March","Missing vendor quotes, with Lee chasing them","Slides for the approval meeting"],"decisions":["The old slide template will be reused"],"action_items":[{"task":"Chase the vendor quotes and send them","assigned_to":"Lee","deadline":"chase tomorrow; send by Friday","status":"open","source_timestamp":null},{"task":"Prepare the slides for the 15 March meeting","assigned_to":"Omar","deadline":"15 March","status":"open","source_timestamp":null},{"task":"Do a dry run of the presentation together","assigned_to":null,"deadline":"12 March","status":"open","source_timestamp":null}],"key_dates":[{"date":"15 March","what":"Budget approval deadline and meeting","owner":null,"timestamp":null},{"date":"tomorrow","what":"Lee chases the vendor quotes","owner":"Lee","timestamp":null},{"date":"Friday","what":"Lee sends the vendor quotes","owner":"Lee","timestamp":null},{"date":"12 March","what":"Group dry run of the presentation","owner":null,"timestamp":null}],"unresolved_issues":["Vendor quotes are still missing"],"next_steps":["Lee to obtain and send the vendor quotes","Omar to prepare the slides for 15 March","Group dry run on 12 March"]}
EXAMPLE>>>"""


def _lang_line(output_language: str) -> str:
    return (
        f"Write every text value in {output_language}. Keep JSON keys and the status values "
        f"in English exactly as shown."
    )


def _checklist_block(date_checklist: str, must_include: str) -> str:
    out = ""
    if date_checklist:
        out += (
            "\n\nDATE / TIME CHECKLIST - an automatic scan found these date and time mentions in "
            "the transcript. Every one that is a deadline, milestone, event or commitment MUST "
            "appear in key_dates, and also in the matching action item or decision where it "
            f"belongs:\n{date_checklist}"
        )
    if must_include:
        out += (
            "\n\nREVISION: an earlier draft of this summary left out the following date mentions. "
            "Produce the complete summary again and make sure each one is included (in key_dates "
            f"and wherever it belongs), keeping everything else that was correct:\n{must_include}"
        )
    return out


def build_chunk_prompt(
    chunk_text: str,
    context_text: str,
    index: int,
    total: int,
    languages: list[str],
    output_language: str,
    date_checklist: str = "",
) -> str:
    langs = ", ".join(languages) if languages else "not labelled (may be mixed)"
    ctx = ""
    if context_text:
        ctx = (
            "The lines below are the END of the previous part, shown only for context. "
            "Do NOT extract items from them again.\n<<<CONTEXT\n"
            f"{context_text}\nCONTEXT>>>\n\n"
        )
    checklist = ""
    if date_checklist:
        checklist = (
            "\n\nDATE / TIME CHECKLIST for this part - put every one that is a deadline, "
            f"milestone, event or commitment into \"dates\":\n{date_checklist}"
        )
    return (
        f"TASK:CHUNK\n"
        f"You are reading part {index + 1} of {total} of a meeting transcript. "
        f"Languages detected: {langs}.\n\n{ctx}"
        f"Extract notes from THIS PART only, as JSON with exactly this shape:\n{CHUNK_SHAPE}\n\n"
        "Later steps see only your notes, not the transcript, so be exhaustive:\n"
        "- every commitment is an action item (a task agreed by the group with no named owner has "
        "assigned_to null; never guess owners; keep the deadline in the speaker's words);\n"
        "- decisions only if clearly agreed;\n"
        "- every date, time, deadline, milestone and promised duration goes into \"dates\";\n"
        "- unresolved = open questions, blockers, pending items, postponed topics;\n"
        "- facts = numbers, names, figures and exact statements worth keeping.\n"
        f"Use empty lists when there is nothing. {_lang_line(output_language)}\n\n"
        f"<<<TRANSCRIPT\n{chunk_text}\nTRANSCRIPT>>>{checklist}"
    )


def build_merge_prompt(notes_json: str, output_language: str) -> str:
    return (
        "TASK:MERGE\n"
        "Below are notes extracted from consecutive parts of one meeting (JSON list, in order). "
        "Merge them into ONE set of notes with exactly the same JSON shape:\n"
        f"{CHUNK_SHAPE}\n\n"
        "Remove duplicates, but keep EVERY distinct decision, action item, date, deadline, number "
        "and name. If an action item or decision was changed later, keep the latest version. "
        "Condense topics (shorter summaries, merge similar topics) so the result is clearly "
        "shorter than the input, without dropping decisions, action items or dates. "
        "Keep chronological order. Do not add anything that is not in the notes. "
        f"{_lang_line(output_language)}\n\n"
        f"<<<NOTES\n{notes_json}\nNOTES>>>"
    )


def build_final_prompt(
    body: str,
    source: str,  # "transcript" | "notes"
    languages: list[str],
    output_language: str,
    prefs: SummaryPreferences,
    speakers: list[str],
    date_checklist: str = "",
    must_include: str = "",
) -> str:
    langs = ", ".join(languages) if languages else "not labelled (may be mixed)"
    spk = ", ".join(speakers) if speakers else "not labelled"
    if source == "transcript":
        intro = f"Below is the full meeting transcript. Languages detected: {langs}. Speakers: {spk}."
        fence = ("<<<TRANSCRIPT", "TRANSCRIPT>>>")
    else:
        intro = (
            "Below are structured notes extracted, in order, from the whole meeting transcript "
            f"(it was too long to read at once). Languages detected: {langs}. Speakers: {spk}. "
            "Treat the notes as the source of truth: carry over every decision, action item and "
            "every entry of their \"dates\" lists (merge duplicates)."
        )
        fence = ("<<<NOTES", "NOTES>>>")
    focus = f"\nPay particular attention to: {prefs.focus}." if prefs.focus else ""
    detail = (
        "Keep it tight."
        if prefs.style == "concise"
        else "Be thorough; include supporting detail for each point."
    )
    return (
        f"TASK:FINAL\n{intro}\n\n"
        f"Produce the final meeting summary as JSON with exactly this shape:\n{FINAL_SHAPE}\n\n"
        f"{FIELD_GUIDE}\n\n"
        f"Use at most {prefs.max_key_points} key discussion points. {detail}{focus}\n"
        f"Use empty lists when nothing applies. {_lang_line(output_language)}\n\n"
        f"{EXAMPLE}\n\n"
        f"{fence[0]}\n{body}\n{fence[1]}"
        f"{_checklist_block(date_checklist, must_include)}\n\n"
        "Now output the JSON for THIS meeting only."
    )


def build_repair_prompt(bad_output: str, error: str) -> str:
    return (
        "TASK:REPAIR\n"
        "Your previous answer was not acceptable.\n"
        f"Problem: {error[:500]}\n\n"
        "Return the corrected answer as ONE valid JSON object with the required keys, "
        "no markdown, no commentary. Previous answer:\n"
        f"<<<PREVIOUS\n{bad_output[:3000]}\nPREVIOUS>>>"
    )