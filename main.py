from __future__ import annotations

import itertools
import math
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# HackerRank Orchestrate - Buy or Wait?
# Deterministic financial decision engine. No API key is required.

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

TERMINAL_BAD = {"failed", "cancelled", "unrealized"}
CASH_STATUSES = {"settled", "scheduled", "pending"}


def norm_date(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def money(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(round(float(x) + 1e-9, 2))


def fmt_money(x: float) -> str:
    return f"{money(x):.2f}".rstrip("0").rstrip(".")


def split_pipe(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}


def amount_from_text(text: str) -> Optional[float]:
    # Used only for optional message/image evidence. Never treats absence as zero.
    if not text:
        return None
    matches = re.findall(r"(?:[$€]|\b(?:INR|IDR|ZAR|USD|EUR)\s*)?([0-9][0-9,]*(?:\.\d+)?)", text, re.I)
    vals = []
    for m in matches:
        try:
            vals.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return max(vals) if vals else None


@dataclass
class Change:
    event_id: str
    action: str
    new_amount: Optional[float]
    monthly_saving: float


class Engine:
    def __init__(self, root: Path):
        self.root = root
        self.dataset = root / "dataset"
        self.requests = pd.read_csv(self.dataset / "requests.csv")
        self.profiles = pd.read_csv(self.dataset / "financial_profiles.csv")
        self.events = pd.read_csv(self.dataset / "financial_events.csv")
        self.options = pd.read_csv(self.dataset / "request_payment_options.csv")
        self.messages = pd.read_csv(self.dataset / "messages.csv") if (self.dataset / "messages.csv").exists() else pd.DataFrame()
        self.images = pd.read_csv(self.dataset / "images.csv") if (self.dataset / "images.csv").exists() else pd.DataFrame()
        self.rates = pd.read_csv(self.dataset / "exchange_rates.csv")

        self.events["event_date"] = pd.to_datetime(self.events["event_date"], errors="coerce").dt.normalize()
        self.events["settlement_date"] = pd.to_datetime(self.events["settlement_date"], errors="coerce").dt.normalize()
        self.requests["request_date"] = pd.to_datetime(self.requests["request_date"]).dt.normalize()
        self.requests["desired_completion_date"] = pd.to_datetime(self.requests["desired_completion_date"]).dt.normalize()
        self.options["first_payment_date"] = pd.to_datetime(self.options["first_payment_date"]).dt.normalize()
        self.messages["sent_at"] = pd.to_datetime(self.messages["sent_at"], errors="coerce") if not self.messages.empty else pd.Series(dtype="datetime64[ns]")
        self.rates["rate_date"] = pd.to_datetime(self.rates["rate_date"]).dt.normalize()

        self.profiles_by_user = self.profiles.set_index("user_id").to_dict("index")
        self.events_by_user = {u: g.copy() for u, g in self.events.groupby("user_id")}
        self.options_by_request = {r: g.copy() for r, g in self.options.groupby("request_id")}
        self.messages_by_user = {u: g.copy() for u, g in self.messages.groupby("user_id")} if not self.messages.empty else {}

        self._apply_message_status_hints()
        self._resolve_image_amounts_if_available()

    # ---------- Evidence / normalization ----------
    def _apply_message_status_hints(self):
        """Use only explicit message amendments; message instructions never override rules."""
        if self.messages.empty:
            return
        for _, m in self.messages.iterrows():
            eid = m.get("related_event_id")
            if pd.isna(eid) or not str(eid).strip():
                continue
            eid = str(eid)
            mask = self.events.event_id.astype(str).eq(eid)
            if not mask.any():
                continue
            text = str(m.get("message_text", "")).lower()
            # Explicit cancellation / reversal takes precedence.
            if any(k in text for k in ["cancelled", "canceled", "reversed", "payment failed"]):
                self.events.loc[mask, "status"] = "cancelled"
            elif any(k in text for k in ["received", "settled", "successfully paid", "payment was received"]):
                # Only amend a supplied event when the message explicitly states settlement.
                self.events.loc[mask, "status"] = "settled"

    def _resolve_image_amounts_if_available(self):
        missing = self.events[self.events.amount.isna()]
        if missing.empty or self.images.empty:
            return
        media = self.dataset / "media" / "images"
        if not media.exists():
            return
        try:
            from PIL import Image
            import pytesseract
        except Exception:
            return

        image_map = self.images.set_index("related_event_id")["image_id"].to_dict()
        for idx, row in missing.iterrows():
            image_id = image_map.get(row.event_id)
            if not image_id:
                continue
            path = media / f"{image_id}.png"
            if not path.exists():
                continue
            try:
                text = pytesseract.image_to_string(Image.open(path))
                amt = amount_from_text(text)
                if amt is not None:
                    self.events.at[idx, "amount"] = amt
            except Exception:
                pass

    def convert(self, amount: float, from_cur: str, to_cur: str, date: pd.Timestamp) -> float:
        if from_cur == to_cur:
            return float(amount)

        date = norm_date(date)

        # Find rates for the currency pair. The supplied hackathon table
        # contains rates on selected dates, not necessarily every day.
        pair = self.rates[
            (self.rates["from_currency"] == from_cur) &
            (self.rates["to_currency"] == to_cur)
        ].copy()

        if not pair.empty:
            pair = pair.sort_values("rate_date")

            # 1. Exact supplied date.
            exact = pair[pair["rate_date"] == date]
            if not exact.empty:
                return float(amount) * float(exact.iloc[0]["rate"])

            # 2. Most recent supplied rate on or before the event date.
            previous = pair[pair["rate_date"] <= date]
            if not previous.empty:
                return float(amount) * float(previous.iloc[-1]["rate"])

            # 3. If the event is earlier than every supplied rate for this
            # pair, use the earliest supplied rate rather than inventing one.
            return float(amount) * float(pair.iloc[0]["rate"])

        # If no direct pair exists, try a multi-hop conversion using the
        # most recent available rate date on or before the event date.
        available_dates = sorted(self.rates["rate_date"].dropna().unique())
        if not available_dates:
            raise ValueError("Exchange-rate table is empty.")

        previous_dates = [d for d in available_dates if d <= date]
        rate_date = previous_dates[-1] if previous_dates else available_dates[0]
        rates = self.rates[self.rates["rate_date"] == rate_date]

        graph = {}
        for _, r in rates.iterrows():
            graph.setdefault(str(r["from_currency"]), []).append(
                (str(r["to_currency"]), float(r["rate"]))
            )

        queue = [(from_cur, 1.0)]
        visited = {from_cur}

        while queue:
            current, factor = queue.pop(0)
            for next_currency, rate in graph.get(current, []):
                if next_currency == to_cur:
                    return float(amount) * factor * rate
                if next_currency not in visited:
                    visited.add(next_currency)
                    queue.append((next_currency, factor * rate))

        raise ValueError(
            f"Missing FX conversion {from_cur}->{to_cur} for {date.date()}"
        )

    # ---------- Event reconstruction ----------
    def user_events(self, user: str) -> pd.DataFrame:
        return self.events_by_user.get(user, pd.DataFrame(columns=self.events.columns)).copy()

    def explicit_future_events(self, user: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        e = self.user_events(user)
        if e.empty:
            return e
        e = e[e.settlement_date.notna() & (e.settlement_date > start) & (e.settlement_date <= end)].copy()
        e = e[~e.status.isin(TERMINAL_BAD)]
        # Pending credits are explicitly excluded by the challenge.
        e = e[~((e.direction == "credit") & (e.status == "pending"))]
        return e

    def recurring_series(self, user: str, asof: pd.Timestamp) -> List[dict]:
        e = self.user_events(user)
        if e.empty:
            return []
        h = e[(e.status == "settled") & e.settlement_date.notna() & (e.settlement_date <= asof)
              & e.amount.notna() & e.direction.isin(["debit", "credit"])].copy()
        series = []
        if h.empty:
            return series
        for key, g in h.groupby(["description", "category", "direction", "event_type"], dropna=False):
            g = g.sort_values("settlement_date")
            dates = g.settlement_date.tolist()
            if len(dates) < 4:
                continue
            diffs = pd.Series(dates).diff().dt.days.dropna().to_numpy()
            if len(diffs) < 3:
                continue
            regular = ((diffs >= 20) & (diffs <= 40)).mean()
            med = float(pd.Series(diffs).median())
            if regular < 0.75 or not (20 <= med <= 40):
                continue
            last = g.iloc[-1]
            series.append({
                "key": key,
                "description": str(last.description),
                "category": str(last.category),
                "direction": str(last.direction),
                "event_type": str(last.event_type),
                "interval": int(round(med)),
                "amount": float(g.tail(3).amount.median()),
                "currency": str(last.currency),
                "last_date": last.settlement_date,
                "event_id": str(last.event_id),
                "flexibility": str(last.flexibility),
                "minimum_allowed_amount": (float(last.minimum_allowed_amount) if pd.notna(last.minimum_allowed_amount) else None),
            })
        return series

    def forecast(self, user: str, request_date: pd.Timestamp, changes: List[Change] | None = None) -> List[Tuple[pd.Timestamp, float, str]]:
        changes = changes or []
        end = request_date + pd.Timedelta(days=90)
        explicit = self.explicit_future_events(user, request_date, end)
        result: List[Tuple[pd.Timestamp, float, str]] = []

        # Explicit future records are authoritative.
        for _, r in explicit.iterrows():
            if pd.isna(r.amount):
                continue  # Never invent an image amount.
            amt = self.convert(float(r.amount), str(r.currency), self.profiles_by_user[user]["home_currency"], r.settlement_date)
            delta = amt if r.direction == "credit" else -amt
            result.append((r.settlement_date, delta, str(r.event_id)))

        # Forecast recurring series only when there is no supplied future event near that occurrence.
        change_map = {c.event_id: c for c in changes}
        for s in self.recurring_series(user, request_date):
            if s["direction"] not in {"debit", "credit"}:
                continue
            cur = s["last_date"] + pd.Timedelta(days=s["interval"])
            while cur <= end:
                exists = explicit[(explicit.description == s["description"])
                                  & (abs((explicit.settlement_date - cur).dt.days) <= 3)]
                if exists.empty:
                    amt = s["amount"]
                    ch = change_map.get(s["event_id"])
                    if ch and s["direction"] == "debit":
                        amt = 0.0 if ch.action == "stop" else float(ch.new_amount or 0.0)
                    home = self.profiles_by_user[user]["home_currency"]
                    amt_home = self.convert(amt, s["currency"], home, cur)
                    delta = amt_home if s["direction"] == "credit" else -amt_home
                    if abs(delta) > 1e-12:
                        result.append((cur, delta, s["event_id"]))
                cur += pd.Timedelta(days=s["interval"])
        return sorted(result, key=lambda x: (x[0], x[2]))

    def min_balance_after(self, user: str, request_date: pd.Timestamp, request_payment: List[Tuple[pd.Timestamp, float]], changes: List[Change] | None = None) -> float:
        p = self.profiles_by_user[user]
        bal = float(p["current_available_balance"])
        events = self.forecast(user, request_date, changes)
        all_items = [(request_date, x, "REQUEST") for _, x in request_payment] + events
        all_items.sort(key=lambda x: (x[0], 0 if x[2] == "REQUEST" else 1))
        min_bal = bal
        for _, delta, _ in all_items:
            bal += delta
            min_bal = min(min_bal, bal)
        return min_bal

    def safe_amount_today(self, user: str, request_date: pd.Timestamp, requested: float) -> float:
        p = self.profiles_by_user[user]
        bal = float(p["current_available_balance"])
        minimum = float(p["minimum_balance_to_keep"])
        events = self.forecast(user, request_date)
        cum = 0.0
        min_cum = 0.0
        for _, delta, _ in events:
            cum += delta
            min_cum = min(min_cum, cum)
        safe = bal + min_cum - minimum
        return money(max(0.0, min(float(requested), safe)))

    def full_safe_on(self, user: str, request_date: pd.Timestamp, amount: float, pay_date: pd.Timestamp, changes: List[Change] | None = None) -> bool:
        p = self.profiles_by_user[user]
        bal = float(p["current_available_balance"])
        minimum = float(p["minimum_balance_to_keep"])
        events = self.forecast(user, request_date, changes)
        all_items = events + [(pay_date, -float(amount), "REQUEST")]
        all_items.sort(key=lambda x: (x[0], 0 if x[2] == "REQUEST" else 1))
        for dt, delta, _ in all_items:
            if dt < request_date:
                continue
            bal += delta
            if bal < minimum - 1e-7:
                return False
        return True

    def earliest_full_date(self, user: str, request_date: pd.Timestamp, amount: float, deadline: pd.Timestamp, changes: List[Change] | None = None) -> Optional[pd.Timestamp]:
        end = request_date + pd.Timedelta(days=90)
        limit = min(deadline, end)
        for i in range((limit - request_date).days + 1):
            d = request_date + pd.Timedelta(days=i)
            if self.full_safe_on(user, request_date, amount, d, changes):
                return d
        return None

    # ---------- Flexible spending changes ----------
    def allowed_changes(self, user: str, request_date: pd.Timestamp) -> List[Change]:
        p = self.profiles_by_user[user]
        reduce_cats = split_pipe(p.get("expense_categories_user_is_willing_to_reduce"))
        stop_cats = split_pipe(p.get("expense_categories_user_is_willing_to_stop"))
        protected = split_pipe(p.get("expense_categories_to_protect"))
        candidates = []
        for s in self.recurring_series(user, request_date):
            if s["direction"] != "debit" or s["category"] in protected:
                continue
            flex = s["flexibility"]
            monthly = self.convert(s["amount"], s["currency"], p["home_currency"], request_date)
            if "stoppable" in flex and s["category"] in stop_cats:
                candidates.append(Change(s["event_id"], "stop", None, monthly))
            if "reducible" in flex and s["category"] in reduce_cats and s["minimum_allowed_amount"] is not None:
                new_home = self.convert(s["minimum_allowed_amount"], s["currency"], p["home_currency"], request_date)
                candidates.append(Change(s["event_id"], "reduce", new_home, max(0.0, monthly - new_home)))
            if "reducible_or_stoppable" in flex:
                if s["category"] in stop_cats:
                    candidates.append(Change(s["event_id"], "stop", None, monthly))
                if s["category"] in reduce_cats and s["minimum_allowed_amount"] is not None:
                    new_home = self.convert(s["minimum_allowed_amount"], s["currency"], p["home_currency"], request_date)
                    candidates.append(Change(s["event_id"], "reduce", new_home, max(0.0, monthly - new_home)))
        # Highest savings first makes enumeration fast and deterministic.
        candidates.sort(key=lambda c: (-c.monthly_saving, c.event_id, c.action))
        return candidates

    def find_change_plan(self, user: str, request_date: pd.Timestamp, amount: float, deadline: pd.Timestamp) -> List[Change] | None:
        candidates = self.allowed_changes(user, request_date)[:12]
        if not candidates:
            return None
        # Enumerate 0..3 changes. More changes are never needed by the output contract.
        for k in range(1, min(3, len(candidates)) + 1):
            best = None
            for combo in itertools.combinations(candidates, k):
                if len({c.event_id for c in combo}) < len(combo):
                    continue
                changes = list(combo)
                d = self.earliest_full_date(user, request_date, amount, deadline, changes)
                if d is None:
                    continue
                # Within same number of changes, prefer lower cost (changes themselves have no fee), then earlier date, then IDs.
                score = (d, tuple(sorted(c.event_id for c in changes)))
                if best is None or score < best[0]:
                    best = (score, changes)
            if best:
                return best[1]
        return None

    # ---------- Payment options ----------
    def option_schedule(self, row) -> List[Tuple[pd.Timestamp, float]]:
        n = int(row.number_of_payments)
        first = norm_date(row.first_payment_date)
        freq = int(row.payment_frequency_days) if pd.notna(row.payment_frequency_days) else 0
        return [(first + pd.Timedelta(days=freq * i), money(float(row.payment_amount))) for i in range(n)]

    def option_allowed(self, user: str, row) -> bool:
        p = self.profiles_by_user[user]
        methods = split_pipe(p.get("payment_methods_user_will_consider"))
        if str(row.payment_method) not in methods:
            return False
        if row.payment_method == "installments":
            max_months = p.get("max_installment_months")
            if pd.isna(max_months) or str(max_months).strip() == "":
                return False
            months = (int(row.number_of_payments) * (int(row.payment_frequency_days) if pd.notna(row.payment_frequency_days) else 30)) / 30.44
            if months > float(max_months) + 1e-9:
                return False
        return True

    def option_safe(self, user: str, request_date: pd.Timestamp, row, deadline: pd.Timestamp, changes: List[Change] | None = None) -> bool:
        schedule = self.option_schedule(row)
        if not schedule or schedule[-1][0] > deadline:
            return False
        p = self.profiles_by_user[user]
        bal = float(p["current_available_balance"])
        minimum = float(p["minimum_balance_to_keep"])
        events = self.forecast(user, request_date, changes)
        items = events + [(d, -a, "REQUEST") for d, a in schedule]
        items.sort(key=lambda x: (x[0], 0 if x[2] == "REQUEST" else 1))
        for d, delta, _ in items:
            if d < request_date:
                continue
            bal += delta
            if bal < minimum - 1e-7:
                return False
        return True

    def format_schedule(self, schedule: List[Tuple[pd.Timestamp, float]]) -> str:
        return "|".join(f"{d.date().isoformat()}:{fmt_money(a)}" for d, a in sorted(schedule))

    # ---------- Decision ----------
    def decide(self, r) -> dict:
        user = str(r.user_id)
        date = norm_date(r.request_date)
        deadline = norm_date(r.desired_completion_date)
        requested = float(r.requested_amount)
        p = self.profiles_by_user[user]
        home = str(p["home_currency"])
        safe_today = self.safe_amount_today(user, date, requested)

        # Financial capacity is independent of payment-method preference.
        earliest = self.earliest_full_date(user, date, requested, deadline)
        options = self.options_by_request.get(str(r.request_id), pd.DataFrame())

        # Spending changes can make otherwise unsafe plans safe.
        change_plan = self.find_change_plan(user, date, requested, deadline)

        candidates = []
        if not options.empty:
            for _, o in options.iterrows():
                if not self.option_allowed(user, o):
                    continue
                if self.option_safe(user, date, o, deadline):
                    sched = self.option_schedule(o)
                    candidates.append({
                        "kind": str(o.payment_method),
                        "schedule": sched,
                        "cost": float(o.total_payable_amount),
                        "start": sched[0][0],
                        "count": len(sched),
                        "option_id": str(o.payment_option_id),
                        "changes": [],
                    })

        # A payment option may become safe after permitted spending changes.
        if not options.empty and change_plan:
            for _, o in options.iterrows():
                if not self.option_allowed(user, o):
                    continue
                if self.option_safe(user, date, o, deadline, change_plan):
                    sched = self.option_schedule(o)
                    candidates.append({
                        "kind": str(o.payment_method),
                        "schedule": sched,
                        "cost": float(o.total_payable_amount),
                        "start": sched[0][0],
                        "count": len(sched),
                        "option_id": str(o.payment_option_id),
                        "changes": change_plan,
                    })

        # Full payment capacity with a spending change, if the user accepts full_payment.
        if "full_payment" in split_pipe(p.get("payment_methods_user_will_consider")) and change_plan:
            if self.full_safe_on(user, date, requested, date, change_plan):
                candidates.append({
                    "kind": "full_payment", "schedule": [(date, money(requested))], "cost": requested,
                    "start": date, "count": 1, "option_id": "~change~", "changes": change_plan,
                })

        # Ranking contract: deadline completion, no changes, lowest total cost, earliest start, fewer payments, option id.
        def rank(c):
            completes = c["schedule"][-1][0] <= deadline
            return (-int(completes), int(len(c["changes"]) > 0), c["cost"], c["start"], c["count"], c["option_id"])

        candidates.sort(key=rank)

        # Determine whether a no-change full payment is safe today.
        full_now = self.full_safe_on(user, date, requested, date)
        accepts_full = "full_payment" in split_pipe(p.get("payment_methods_user_will_consider"))

        # If full payment today is both safe and accepted, it is the cleanest outcome.
        if full_now and accepts_full:
            return self._row(r, requested, "affordable_now", "full_payment", [(date, requested)], date, [], home,
                             f"Pay {home} {fmt_money(requested)} today. This keeps at least {home} {fmt_money(float(p['minimum_balance_to_keep']))} available over the 90-day forecast.")

        # Prefer a safe eligible option. If it is an installment plan, use its exact supplied schedule.
        if candidates:
            c = candidates[0]
            method = c["kind"]
            changes = c["changes"]
            sched = c["schedule"]
            status = "affordable_with_plan"
            if method == "full_payment" and sched[0][0] > date:
                status = "affordable_later"
                method = "wait"
            elif method == "full_payment" and sched[0][0] == date and not changes:
                status = "affordable_now"
            change_text = self.format_changes(changes)
            return self._row(r, safe_today, status, method, sched, earliest or sched[-1][0], changes, home,
                             self.explanation(r, home, requested, safe_today, status, method, sched, changes, p))

        # Partial payment is a special two-payment schedule and doesn't need a supplied option.
        accepts_partial = "partial_payment" in split_pipe(p.get("payment_methods_user_will_consider"))
        if bool(r.allows_partial_payment) and accepts_partial and 0 < safe_today < requested:
            # Remaining amount is paid on the earliest safe date for the remaining balance.
            remaining = money(requested - safe_today)
            d2 = None
            for i in range((deadline - date).days + 1):
                d = date + pd.Timedelta(days=i)
                if self.full_safe_on(user, date, remaining, d):
                    # Check the actual two-payment plan, not merely the remaining payment.
                    if self.plan_safe(user, date, [(date, safe_today), (d, remaining)]):
                        d2 = d
                        break
            if d2 is not None and d2 <= deadline:
                sched = [(date, safe_today), (d2, remaining)]
                return self._row(r, safe_today, "affordable_with_plan", "partial_payment", sched, d2, [], home,
                                 f"Pay {home} {fmt_money(safe_today)} today and the remaining {home} {fmt_money(remaining)} on {d2.date().isoformat()}. This completes the request by the deadline while preserving the minimum balance.")

        # If full amount becomes safe later by the deadline, wait (only when full_payment is accepted).
        if earliest is not None and accepts_full:
            return self._row(r, safe_today, "affordable_later", "wait", [(earliest, requested)], earliest, [], home,
                             f"Wait until {earliest.date().isoformat()}, when {home} {fmt_money(requested)} is forecast safe in full. Paying earlier would put the {home} {fmt_money(float(p['minimum_balance_to_keep']))} minimum at risk.")

        # If a change plan exists but couldn't make an accepted payment method safe, do not recommend it.
        return self._row(r, safe_today, "not_affordable", "not_recommended", [], None, [], home,
                         f"Do not make this payment by {deadline.date().isoformat()}. None of the eligible options keeps the {home} {fmt_money(float(p['minimum_balance_to_keep']))} minimum protected.")

    def plan_safe(self, user, request_date, schedule):
        p = self.profiles_by_user[user]
        bal = float(p["current_available_balance"])
        minimum = float(p["minimum_balance_to_keep"])
        items = self.forecast(user, request_date) + [(d, -a, "REQUEST") for d, a in schedule]
        items.sort(key=lambda x: (x[0], 0 if x[2] == "REQUEST" else 1))
        for d, delta, _ in items:
            if d < request_date:
                continue
            bal += delta
            if bal < minimum - 1e-7:
                return False
        return True

    def format_changes(self, changes: List[Change]) -> str:
        if not changes:
            return "none"
        out = []
        for c in changes:
            if c.action == "stop":
                out.append(f"stop:{c.event_id}")
            else:
                out.append(f"reduce_to:{c.event_id}:{fmt_money(c.new_amount or 0)}")
        return "|".join(out)

    def explanation(self, r, home, requested, safe_today, status, method, sched, changes, p):
        change_text = self.format_changes(changes)
        if changes:
            return (f"Use {method.replace('_',' ')} after the permitted spending changes ({change_text}). "
                    f"This completes {home} {fmt_money(requested)} by {sched[-1][0].date().isoformat()} while keeping at least "
                    f"{home} {fmt_money(float(p['minimum_balance_to_keep']))} available.")
        if method == "installments":
            return (f"Use the supplied installment option with {len(sched)} payments starting {sched[0][0].date().isoformat()}. "
                    f"The schedule completes the {home} {fmt_money(requested)} request by the deadline without dropping below the minimum balance.")
        return (f"Use {method.replace('_',' ')} to complete {home} {fmt_money(requested)} while maintaining the "
                f"{home} {fmt_money(float(p['minimum_balance_to_keep']))} minimum.")

    def _row(self, r, amount_safe, status, method, schedule, earliest, changes, home, explanation):
        return {
            "request_id": r.request_id,
            "amount_safe_to_pay": money(max(0, min(float(r.requested_amount), amount_safe))),
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": self.format_schedule(schedule) if schedule else "none",
            "earliest_date_for_full_payment": earliest.date().isoformat() if earliest is not None else "",
            "spending_changes_needed": self.format_changes(changes),
            "decision_explanation": explanation,
        }

    def validate(self, row, request):
        amt = float(row["amount_safe_to_pay"])
        req = float(request.requested_amount)
        assert -1e-8 <= amt <= req + 1e-8
        assert row["affordability_status"] in {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
        assert row["recommended_payment_method"] in {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
        if row["recommended_payment_method"] == "partial_payment":
            parts = row["payment_plan"].split("|")
            assert len(parts) == 2
        if row["recommended_payment_method"] == "not_recommended":
            assert row["payment_plan"] == "none"

    def run(self) -> pd.DataFrame:
        rows = []
        for _, r in self.requests.iterrows():
            try:
                row = self.decide(r)
                self.validate(row, r)
            except Exception as exc:
                # Deterministic safe fallback: never invent a positive recommendation when data is incomplete.
                user = str(r.user_id)
                p = self.profiles_by_user[user]
                row = self._row(r, self.safe_amount_today(user, norm_date(r.request_date), float(r.requested_amount)),
                                "not_affordable", "not_recommended", [], None, [], str(p["home_currency"]),
                                f"Do not make this payment by {norm_date(r.desired_completion_date).date().isoformat()}. Financial evidence is incomplete for a safe recommendation.")
            rows.append(row)
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        return out


def main():
    root = Path(__file__).resolve().parents[1]
    engine = Engine(root)
    out = engine.run()
    target = root / "output.csv"
    out.to_csv(target, index=False)
    print(f"Wrote {len(out)} predictions to {target}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
