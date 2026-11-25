"""Skilvanta demo MCP server. FICTIONAL company, SYNTHETIC data.

In a real deployment the company's backend runs this server next to its own database, and
the brain only ever sees what these tools return. Here a few made-up learners live in memory.

Run it:
    python mcp_server/server.py                      # stdio: the brain starts it itself
    python mcp_server/server.py --http --port 8792   # Streamable HTTP at /mcp

Until the access layer forwards the user's token (milestone M7), every call acts as one demo
learner, chosen with SKILVANTA_LEARNER_ID (default L-1042).
"""

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# The demo is pinned to one date so its answers, and the eval cases, stay stable.
DEMO_TODAY = date(2026, 11, 25)
SESSIONS_PER_MONTH = 2  # matches docs/mentorship-sessions.md
DEFAULT_LEARNER = "L-1042"


@dataclass(frozen=True)
class Course:
    id: str
    name: str


COURSES = {
    c.id: c
    for c in (
        Course("FSWD", "Full-Stack Web Development"),
        Course("DSML", "Data Science and Machine Learning"),
        Course("BEPY", "Backend Engineering with Python"),
        Course("DSA", "DSA and Interview Prep"),
    )
}

LEARNERS: dict[str, dict[str, Any]] = {
    "L-1042": {
        "name": "Riya Kulkarni",
        "mentor": "Arjun Desai",
        "sessions_used": {"2026-11": ["2026-11-12"]},
        "enrollments": {
            "BEPY": {
                "batch": "BEPY-2026-09",
                "status": "active",
                "start_date": "2026-09-05",
                "end_date": "2026-12-27",
                "payment": "No-cost EMI over 6 months: 3 of 6 instalments of Rs 10,000 paid",
                "progress": {
                    "modules_completed": 9,
                    "modules_total": 16,
                    "current_module": "Module 10: Caching with Redis",
                    "live_classes_attended": 22,
                    "live_classes_held": 25,
                    "pending_assignments": [
                        {"title": "Assignment 7: Rate-limited REST API", "due": "2026-11-29"},
                        {"title": "Assignment 8: Background jobs with Celery", "due": "2026-12-06"},
                    ],
                },
            },
            "DSA": {
                "batch": "self-paced",
                "status": "active",
                "start_date": "2026-10-01",
                "end_date": None,
                "payment": "Paid in full: Rs 30,000",
                "progress": {
                    "weeks_completed": 4,
                    "weeks_total": 10,
                    "current_week": "Week 5: Graphs",
                    "doubt_sessions_attended": 3,
                    "pending_assignments": [
                        {"title": "Mock interview 2", "due": "not booked yet"},
                    ],
                },
            },
        },
    },
    "L-1077": {
        "name": "Kabir Shah",
        "mentor": "Meera Iyer",
        "sessions_used": {"2026-11": ["2026-11-04", "2026-11-18"]},
        "enrollments": {
            "DSML": {
                "batch": "DSML-2026-08",
                "status": "paused",
                "start_date": "2026-08-11",
                "end_date": "2027-02-22",
                "payment": "Paid in full: Rs 95,000",
                "progress": {
                    "modules_completed": 6,
                    "modules_total": 20,
                    "current_module": "Module 7: Linear models",
                    "live_classes_attended": 30,
                    "live_classes_held": 41,
                    "pending_assignments": [],
                },
            },
        },
    },
}

MENTOR_SLOTS: dict[str, list[str]] = {  # IST, ISO 8601
    "Arjun Desai": [
        "2026-11-27T19:00", "2026-11-29T11:00", "2026-12-02T20:00", "2026-12-04T19:30",
        "2026-12-09T19:00",
    ],
    "Meera Iyer": ["2026-11-26T18:00", "2026-12-01T18:00", "2026-12-08T18:00"],
}  # fmt: skip


def _learner() -> tuple[str, dict[str, Any]]:
    learner_id = os.environ.get("SKILVANTA_LEARNER_ID", DEFAULT_LEARNER)
    if learner_id not in LEARNERS:
        raise ToolError(f"No learner account found for id {learner_id}.")
    return learner_id, LEARNERS[learner_id]


def _find_course(query: str) -> Course | None:
    wanted = query.strip().casefold()
    for course in COURSES.values():
        if wanted in (course.id.casefold(), course.name.casefold()):
            return course
    matches = [c for c in COURSES.values() if wanted and wanted in c.name.casefold()]
    return matches[0] if len(matches) == 1 else None


def build_server() -> MCPServer:
    server = MCPServer(
        "skilvanta",
        instructions="Skilvanta learner data: enrollments, course progress, mentor slots.",
        log_level="WARNING",  # expected tool errors (e.g. "not enrolled") are not news
    )

    @server.tool()
    def get_my_enrollments() -> dict[str, Any]:
        """List the courses the signed-in learner is enrolled in, with batch, status
        (active, paused or completed), start and end dates, and fee payment status."""
        learner_id, learner = _learner()
        return {
            "learner_id": learner_id,
            "name": learner["name"],
            "enrollments": [
                {"course_id": cid, "course": COURSES[cid].name}
                | {k: v for k, v in e.items() if k != "progress"}
                for cid, e in learner["enrollments"].items()
            ],
        }

    @server.tool()
    def get_course_progress(course: str) -> dict[str, Any]:
        """Progress of the signed-in learner in one of their courses: modules or weeks
        completed, current module, live-class attendance and pending assignments with due
        dates. `course` is a course id (FSWD, DSML, BEPY, DSA) or the course name."""
        _, learner = _learner()
        found = _find_course(course)
        if found is None:
            known = ", ".join(f"{c.id} ({c.name})" for c in COURSES.values())
            raise ToolError(f"Unknown course '{course}'. Courses: {known}.")
        enrollment = learner["enrollments"].get(found.id)
        if enrollment is None:
            enrolled = ", ".join(COURSES[c].name for c in learner["enrollments"]) or "none"
            raise ToolError(
                f"The learner is not enrolled in {found.name}. Enrolled courses: {enrolled}."
            )
        return {"course": found.name, "status": enrollment["status"], **enrollment["progress"]}

    @server.tool()
    def list_mentor_slots(days_ahead: int = 7) -> dict[str, Any]:
        """Open slots on the signed-in learner's mentor's calendar for the next `days_ahead`
        days (1 to 14), in IST, and how many 1:1 sessions the learner has left this month."""
        if not 1 <= days_ahead <= 14:
            raise ToolError("days_ahead must be between 1 and 14.")
        _, learner = _learner()
        mentor = learner["mentor"]
        until = DEMO_TODAY + timedelta(days=days_ahead)
        slots = [
            s
            for s in MENTOR_SLOTS.get(mentor, [])
            if DEMO_TODAY < datetime.fromisoformat(s).date() <= until
        ]
        month = DEMO_TODAY.strftime("%Y-%m")
        used = len(learner["sessions_used"].get(month, []))
        return {
            "today": DEMO_TODAY.isoformat(),
            "mentor": mentor,
            "open_slots_ist": slots,
            "sessions_left_this_month": max(SESSIONS_PER_MONTH - used, 0),
            "note": "Slots in a new month count against that month's two sessions.",
        }

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP, not stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    server = build_server()
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
