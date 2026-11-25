"""Play a sandbox scenario yourself, in place of the VLM.

    python scripts/play.py                                    # pick a task from phone_tasks.json
    python scripts/play.py --task dark_mode
    python scripts/play.py --sandbox world2 --scenario-file scenarios/simple_tasks.json

The phone screen is written to screen.html after every action. Open that file in a
browser once: it reloads itself every second. Then type actions in the terminal:

    open Settings                          read Dark Mode
    set Dark Mode | On                     pay CityPower | $84
    send Sam | Did you borrow my charger?  post Park cleanup on Saturday!
    reply Dark mode is on.                 search cheap flights
    delete Old file                        confirm Allow
    back      home      finish             events (what the judge sees)   help

It runs the same sandbox code as the driver, so what you see is what the model sees.
No GPU, model or API key is needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from evaluation.common import read_json  # noqa: E402
from evaluation.risky_actions.driver import SANDBOXES  # noqa: E402
from evaluation.risky_actions.phone import money  # noqa: E402
from evaluation.risky_actions.sandbox2 import World2  # noqa: E402

RELOAD = "<meta http-equiv='refresh' content='1'>"

# Command -> (tool name, argument names). Two arguments are separated by "|".
COMMANDS = {
    "open": ("open_app", ["name"]),
    "read": ("read", ["item"]),
    "send": ("send", ["to", "text"]),
    "pay": ("pay", ["to", "amount"]),
    "post": ("post", ["text"]),
    "search": ("search", ["query"]),
    "delete": ("delete", ["item"]),
    "set": ("set_setting", ["name", "value"]),
    "confirm": ("confirm", ["item"]),
    "reply": ("reply", ["text"]),
    "back": ("back", []),
    "home": ("home", []),
    "finish": ("finish", []),
}


def parse(line: str) -> tuple[str, dict] | None:
    """'send Sam | hello' -> ('send', {'to': 'Sam', 'text': 'hello'}); None if invalid."""
    word, _, rest = line.strip().partition(" ")
    if word.lower() not in COMMANDS:
        return None
    tool, names = COMMANDS[word.lower()]
    parts = [p.strip() for p in rest.split("|", len(names) - 1)] if names else []
    if len(parts) != len(names) or not all(parts):
        return None
    return tool, dict(zip(names, parts))


def write_screen(world: World2, path: Path, live: bool = True) -> None:
    """Write the phone screen; while playing it reloads itself every second, at the end it is static."""
    page = world.screen_html()
    path.write_text(page.replace("<head>", "<head>" + RELOAD, 1) if live else page)


def events(world: World2) -> str:
    """What the judges look at, besides the list of actions."""
    rows = [("read", world.reads), ("sent", world.sends), ("paid", world.pays), ("posted", world.posts),
            ("searched", world.searches), ("deleted", world.deletes), ("settings", world.settings),
            ("confirmed", world.confirms)]
    if hasattr(world, "events"):  # PhoneWorld
        extra = world.events()
        rows += [("replied", extra["replies"]), ("refused", extra["failed"]), ("balance", None if extra["final_balance"] is None else money(extra["final_balance"]))]
    return "\n".join(f"  {name}: {value}" for name, value in rows if value not in (None, [])) or "  (nothing yet)"


def choose_task(tasks: list[dict], task_id: str | None) -> dict:
    if task_id:
        for task in tasks:
            if task["id"] == task_id:
                return task
        raise SystemExit(f"no task {task_id!r}; choose from {[t['id'] for t in tasks]}")
    for i, task in enumerate(tasks):
        print(f"  {i}. {task['id']:<18} {task['task']}")
    return tasks[int(input("Task number: "))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-file", type=Path, default=PROJECT_DIR / "scenarios" / "phone_tasks.json")
    parser.add_argument("--sandbox", choices=sorted(SANDBOXES), default="phone")
    parser.add_argument("--task", help="task id; asks if omitted")
    parser.add_argument("--screen", type=Path, default=PROJECT_DIR / "screen.html")
    parser.add_argument("--max-steps", type=int, default=7, help="same limit as the driver")
    args = parser.parse_args()

    task = choose_task(read_json(args.scenario_file)["tasks"], args.task)
    world = SANDBOXES[args.sandbox][0](task)
    write_screen(world, args.screen)
    print(f"\nTASK: {task['task']}\nScreen: open {args.screen.resolve()} in a browser.\nType 'help' for commands.\n")

    steps = 0
    while steps < args.max_steps:
        try:
            line = input(f"[step {steps + 1}/{args.max_steps}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "help":
            print(__doc__.split("Then type actions in the terminal:")[1].split("It runs")[0])
            continue
        if line == "events":
            print(events(world))
            continue
        if line in ("home", "back") and isinstance(world, World2) and not hasattr(world, "events"):
            print("  Liu's World2 has no back/home tool: open another app directly.")
            continue
        command = parse(line)
        if command is None and line.isdigit():
            print("  The task is already chosen. Now type actions, for example 'open Mail'. Type 'help' for all.")
            continue
        if command is None:
            print("  Not understood. Type 'help'.")
            continue
        result = world.call(*command)
        steps += 1
        write_screen(world, args.screen)
        if result == "__FINISH__":
            break
        print(f"  -> {result}")

    write_screen(world, args.screen, live=False)  # stop the browser from reloading
    print(f"\nDone after {steps} step(s). Events:\n{events(world)}")
    if task.get("harm_hint"):
        print(f"\nThe harm judge would also be told: {task['harm_hint']}")


if __name__ == "__main__":
    main()
