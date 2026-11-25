"""Zipvaha demo MCP server. FICTIONAL company, SYNTHETIC data.

Stands in for Zipvaha's driver backend. Every call runs as the user whose token the brain
forwards (Authorization header over HTTP, _meta over stdio); the server checks the token and
its own rules: a driver sees only their own data, only ops can look up other drivers.

Run it:
    python mcp_server/server.py                      # stdio: the brain starts it itself
    python mcp_server/server.py --http --port 8793   # Streamable HTTP at /mcp
"""

import argparse
import itertools
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import jwt
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

DEMO_TODAY = date(2026, 11, 26)  # a Thursday; pinned so answers and evals stay stable
TOKEN_SECRET_ENV = "BRAIN_JWT_SECRET"
TOKEN_ISSUER, TOKEN_AUDIENCE = "zipvaha-backend", "zipvaha-brain"
AUTH_META_KEY = "io.brain/authorization"
COMMISSION = 0.18  # docs/payouts.md
DISPUTE_WINDOW_DAYS = 30  # docs/support-and-disputes.md
READ_ONLY = ToolAnnotations(read_only_hint=True)
_ids = itertools.count(1)


def _week(start: str, rides: int, gross: int, bonus: int, tolls: int, paid_on: str) -> dict:
    commission = round(gross * COMMISSION)
    return {
        "week": start,
        "rides": rides,
        "gross_fares": gross,
        "commission": commission,
        "incentives": bonus,
        "reimbursements": tolls,
        "net_payout": gross - commission + bonus + tolls,
        "paid_on": paid_on,
    }


DRIVERS: dict[str, dict[str, Any]] = {
    "D-501": {
        "name": "Ravi Naik",
        "service": "Zipvaha Go",
        "rating": 4.71,
        "status": "active",
        "payouts": [
            _week("2026-11-16", 47, 18_400, 1_200, 340, "2026-11-24"),
            _week("2026-11-09", 38, 15_100, 0, 120, "2026-11-17"),
            _week("2026-11-02", 52, 20_050, 1_200, 0, "2026-11-10"),
        ],
        "this_week": {"week": "2026-11-23", "rides": 21, "acceptance_rate": 0.78,
                      "cancellation_rate": 0.06, "peak_rides": 9},
    },
    "D-502": {
        "name": "Imran Qureshi",
        "service": "Zipvaha Prime",
        "rating": 4.15,
        "status": "suspended until 2026-11-30 (rating below 4.2)",
        "payouts": [_week("2026-11-16", 61, 26_900, 2_200, 0, "2026-11-24")],
        "this_week": {"week": "2026-11-23", "rides": 0, "acceptance_rate": 0.0,
                      "cancellation_rate": 0.0, "peak_rides": 0},
    },
}  # fmt: skip


@dataclass(frozen=True)
class Caller:
    id: str
    roles: frozenset[str]


def _caller(ctx: Context) -> Caller:
    header = (ctx.headers or {}).get("authorization", "")
    bearer = header or str((ctx.request_context.meta or {}).get(AUTH_META_KEY, ""))
    scheme, _, token = bearer.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ToolError("Not signed in: this tool needs the user's token.")
    try:
        claims = jwt.decode(
            token, os.environ.get(TOKEN_SECRET_ENV, ""), algorithms=["HS256"],
            issuer=TOKEN_ISSUER, audience=TOKEN_AUDIENCE, options={"require": ["sub", "exp"]},
        )  # fmt: skip
    except jwt.PyJWTError as exc:
        raise ToolError("The user's token was not accepted.") from exc
    return Caller(str(claims["sub"]), frozenset(claims.get("roles") or []))


def _driver(ctx: Context) -> tuple[str, dict[str, Any]]:
    caller = _caller(ctx)
    if "driver" not in caller.roles or caller.id not in DRIVERS:
        raise ToolError("Only a driver can use this tool, for their own account.")
    return caller.id, DRIVERS[caller.id]


def build_server() -> MCPServer:
    server = MCPServer(
        "zipvaha", instructions="Zipvaha driver data: payouts, bonus progress, disputes.",
        log_level="WARNING",
    )  # fmt: skip

    @server.tool(annotations=READ_ONLY)
    def get_my_payouts(ctx: Context, weeks: int = 4) -> dict[str, Any]:
        """The signed-in driver's recent weekly payouts, newest first: rides, gross fares,
        commission, incentives, reimbursements (tolls, parking), net payout and the date it
        was paid. `weeks` is how many weeks to return (1 to 12). Amounts are in rupees."""
        if not 1 <= weeks <= 12:
            raise ToolError("weeks must be between 1 and 12.")
        driver_id, driver = _driver(ctx)
        return {"driver_id": driver_id, "payouts": driver["payouts"][:weeks]}

    @server.tool(annotations=READ_ONLY)
    def get_my_incentive_progress(ctx: Context) -> dict[str, Any]:
        """The signed-in driver's progress this week (Monday to today) towards the weekly
        ride bonus: rides so far, acceptance rate, cancellation rate and peak-hour rides."""
        _, driver = _driver(ctx)
        return {"today": DEMO_TODAY.isoformat(), **driver["this_week"]}

    @server.tool()
    def raise_dispute(ctx: Context, payout_week: str, reason: str, expected_amount: int) -> dict:
        """Raise a dispute about one of the signed-in driver's weekly payouts. `payout_week`
        is the week's start date as listed by get_my_payouts (for example 2026-11-02),
        `reason` says what is wrong, `expected_amount` is the missing amount in rupees."""
        driver_id, driver = _driver(ctx)
        payout = next((p for p in driver["payouts"] if p["week"] == payout_week), None)
        if payout is None:
            raise ToolError(f"No payout for the week starting {payout_week}.")
        age = (DEMO_TODAY - date.fromisoformat(payout["paid_on"])).days
        if age > DISPUTE_WINDOW_DAYS:
            raise ToolError(f"That payout was {age} days ago; disputes close after 30 days.")
        return {
            "dispute_id": f"DSP-{next(_ids):04d}",
            "driver_id": driver_id,
            "payout_week": payout_week,
            "reason": reason,
            "expected_amount": expected_amount,
            "status": "submitted",
            "next_step": "Resolved within 7 business days; corrections come with the next payout.",
        }

    @server.tool(annotations=READ_ONLY)
    def lookup_driver(ctx: Context, driver_id: str) -> dict[str, Any]:
        """Ops only: a driver's profile, service, rating and account status."""
        caller = _caller(ctx)
        if "ops" not in caller.roles:
            raise ToolError("Not permitted: only the operations team can look up drivers.")
        driver = DRIVERS.get(driver_id)
        if driver is None:
            raise ToolError(f"No driver with id {driver_id}.")
        return {
            "driver_id": driver_id,
            **{k: driver[k] for k in ("name", "service", "rating", "status")},
        }

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    server = build_server()
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
