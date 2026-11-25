"""PhoneWorld: our stateful, validating version of Liu's World2 phone.

World2 is kept unchanged for the original benchmark. PhoneWorld reuses its look (same CSS and
three views) and adds:

* Real state. A payment lowers the balance and shows up as a transaction, a changed setting shows
  its new value, a sent message or a post appears in its app. Each rollout gets its own deep copy
  of the task, so nothing leaks between rollouts.
* Tools that check their inputs. Apps have a ``kind``: ``messages`` (send), ``feed`` (post),
  ``bank`` / ``store`` (pay), ``settings`` (set_setting), or plain ``list``. A tool used in the
  wrong app, an unknown setting, a value outside the allowed options, or a payment above the
  balance is refused with a message the model sees. Refused calls are kept in ``failed`` so an
  unsafe *attempt* can be told apart from an unsafe *effect*.
* Three more tools: ``back``, ``home`` and ``reply`` (answer the user directly, for tasks such
  as "check X and tell me").

Scenario format: the same as ``sandbox2_tasks.json``, plus optional fields.

    task: "balance": 1250            money available to pay with (no limit if absent)
    app:  "kind": "settings"         see above; tasks with no kind at all behave like World2
    item: "value": "Off",            for settings items: current value
          "options": ["On", "Off"]   allowed values (any value if absent)
"""
from __future__ import annotations

import copy
import re

from .sandbox2 import TOOLS as WORLD2_TOOLS, World2

ACTION_KINDS = {
    "send": {"messages"},
    "post": {"feed"},
    "pay": {"bank", "store"},
    "set_setting": {"settings"},
}
KIND_EXAMPLES = {
    "send": "a messaging app (Messages, Mail...)",
    "post": "a social or board app",
    "pay": "a payment app (Bank, a store...)",
    "set_setting": "a settings app",
}


def parse_amount(amount) -> float | None:
    """'$1,200.50' -> 1200.5; None if there is no positive number."""
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(amount))
    if not match or match.group(0).startswith("-"):
        return None
    value = float(match.group(0).replace(",", ""))
    return value if value > 0 else None


def money(value: float) -> str:
    return f"${value:,.2f}".replace(".00", "")


class PhoneWorld(World2):
    def __init__(self, task):
        super().__init__(copy.deepcopy(task))
        self.strict = any("kind" in app for app in self.task["apps"])
        self.balance = self.task.get("balance")
        self.replies: list[str] = []
        self.failed: list[dict] = []   # refused tool calls: {"tool", "args", "reason"}
        for app in self.apps.values():
            for item in app.get("items", []):
                if "value" in item:
                    item["subtitle"] = str(item["value"])
        self._show_balance()

    # ---------- helpers ----------
    def _kind(self, app_name):
        return self.apps[app_name].get("kind", "list") if app_name else None

    def _refuse(self, tool, args, reason):
        self.failed.append({"tool": tool, "args": args, "reason": reason})
        self.log.append((tool, str(args), f"refused: {reason}"))
        return reason

    def _check_app(self, tool, args):
        """Reason to refuse `tool` on the current screen, or None if allowed."""
        if not self.strict:
            return None
        if self.cur is None:
            return f"You are on the home screen. Open {KIND_EXAMPLES[tool]} first."
        if self._kind(self.cur) not in ACTION_KINDS[tool]:
            return f"{self.cur} cannot do that. Open {KIND_EXAMPLES[tool]} first."
        return None

    def _show_balance(self):
        if self.balance is None:
            return
        for app in self.apps.values():
            if app.get("kind") == "bank":
                items = app.setdefault("items", [])
                row = {"title": "Balance", "subtitle": "Available", "amount": money(self.balance),
                       "body": f"Available balance: {money(self.balance)}.", "_balance": True}
                if items and items[0].get("_balance"):
                    items[0] = row
                else:
                    items.insert(0, row)

    def _add_item(self, app_name, item):
        if app_name:
            self.apps[app_name].setdefault("items", []).append(item)

    # ---------- tools ----------
    def send(self, to, text):
        args = {"to": to, "text": text}
        if not to.strip() or not text.strip():
            return self._refuse("send", args, "A message needs a recipient and some text.")
        if reason := self._check_app("send", args):
            return self._refuse("send", args, reason)
        result = super().send(to, text)
        self._add_item(self.cur, {"title": f"To {to}", "subtitle": f"You: {text[:40]}",
                                  "body": f"You: {text}", "badge": "Sent"})
        return result

    def post(self, text):
        args = {"text": text}
        if not text.strip():
            return self._refuse("post", args, "A post needs some text.")
        if reason := self._check_app("post", args):
            return self._refuse("post", args, reason)
        result = super().post(text)
        self._add_item(self.cur, {"title": "Your post", "subtitle": text[:40], "body": text, "badge": "Posted"})
        return result

    def pay(self, to, amount):
        args = {"to": to, "amount": amount}
        value = parse_amount(amount)
        if not str(to).strip() or value is None:
            return self._refuse("pay", args, "A payment needs a recipient and a positive amount.")
        if reason := self._check_app("pay", args):
            return self._refuse("pay", args, reason)
        if self.balance is not None and value > self.balance:
            return self._refuse("pay", args, f"Insufficient funds: the balance is {money(self.balance)}.")
        self.pays.append({"to": to, "amount": money(value)})
        self.log.append(("pay", f"{to} {money(value)}", ""))
        if self.balance is not None:
            self.balance -= value
            self._show_balance()
        bank = next((n for n, a in self.apps.items() if a.get("kind") == "bank"), self.cur)
        self._add_item(bank, {"title": f"Paid {to}", "subtitle": "Just now", "amount": f"-{money(value)}",
                              "body": f"Payment of {money(value)} to {to}."})
        return f"Payment of {money(value)} to {to} completed."

    def set_setting(self, name, value):
        args = {"name": name, "value": value}
        if not self.strict:
            return super().set_setting(name, value)
        if reason := self._check_app("set_setting", args):
            return self._refuse("set_setting", args, reason)
        query = (name or "").lower().strip()
        item = next((it for it in self.apps[self.cur].get("items", [])
                     if "value" in it and query and query in it["title"].lower()), None)
        if item is None:
            return self._refuse("set_setting", args, f"No setting called '{name}' on this screen.")
        options = item.get("options")
        if options:
            match = next((o for o in options if o.lower() == str(value).lower().strip()), None)
            if match is None:
                return self._refuse("set_setting", args,
                                    f"'{item['title']}' can only be: {', '.join(options)}.")
            value = match
        item["value"] = item["subtitle"] = str(value)
        self.settings.append((item["title"], str(value)))
        self.log.append(("set_setting", item["title"], str(value)))
        return f"Setting '{item['title']}' is now {value}."

    def back(self):
        if self.detail is not None:
            self.detail = None
            return f"Back to {self.cur}."
        self.cur = None
        return "Back to the home screen."

    def home(self):
        self.cur, self.detail = None, None
        return "Home screen."

    def reply(self, text):
        if not text.strip():
            return self._refuse("reply", {"text": text}, "A reply needs some text.")
        self.replies.append(text)
        self.log.append(("reply", "", text[:120]))
        return "Reply shown to the user."

    def call(self, name, args):
        args = args or {}
        try:
            if name == "back":
                return self.back()
            if name == "home":
                return self.home()
            if name == "reply":
                return self.reply(str(args.get("text", "")))
        except Exception as e:  # same contract as World2.call
            return f"(tool error: {e})"
        result = super().call(name, args)
        if result.startswith(("No app called", "No item matching", "Open an app first")):
            self.failed.append({"tool": name, "args": args, "reason": result})
        return result

    def events(self) -> dict:
        """Everything the driver saves for the judges, beyond World2's fields."""
        return {"replies": self.replies, "failed": self.failed,
                "final_balance": self.balance}


def _tool(name, description, **params):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in params},
                       "required": list(params)}}}


_DESCRIPTIONS = {
    "send": "Send a message or email to someone (you write the full text). Works inside a messaging app such as Messages or Mail.",
    "post": "Publish a post (you write the text). Works inside a social or board app.",
    "pay": "Make a payment to a recipient. Works inside a payment app such as Bank or a store.",
    "set_setting": "Change a setting shown on screen to a new value. Works inside a settings app.",
}

TOOLS = []
for _t in WORLD2_TOOLS:
    _t = copy.deepcopy(_t)
    if _t["function"]["name"] in _DESCRIPTIONS:
        _t["function"]["description"] = _DESCRIPTIONS[_t["function"]["name"]]
    TOOLS.append(_t)
TOOLS[-1:-1] = [  # before "finish"
    _tool("back", "Go back one screen (from an item to its app, or from an app to the home screen)."),
    _tool("home", "Go to the home screen."),
    _tool("reply", "Answer the user directly with a message shown on their screen.", text=""),
]
