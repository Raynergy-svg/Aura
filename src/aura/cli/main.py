"""Aura CLI entry point.

Usage:
    aura              # Interactive mode (default — just talk)
    aura --demo       # Run the demo scenario (Maya career decision)
    aura --status     # Show bridge status and exit
    aura --theme NAME # Start with a specific theme
    aura --help       # Show all options
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.aura.cli.companion import AuraCompanion
from src.aura.cli.brand import get_brand, reset_brand, THEMES, RESET

logger = logging.getLogger(__name__)


def run_interactive(companion: AuraCompanion) -> None:
    """Run the interactive conversation loop."""
    brand = get_brand()
    companion.start_session()

    print(brand.render_session_start())

    while True:
        try:
            prompt = brand.render_user_prompt()
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        response = companion.process_input(user_input)

        if response == "__QUIT__":
            break

        is_command = user_input.strip().startswith("/")

        if is_command:
            # Commands render their own formatting — print directly
            print()
            print(response)
            print()
        else:
            # Conversational responses — Aura prefix + wrapped body
            prefix = brand.render_aura_prefix()
            formatted = brand.format_response(response)
            print(f"{prefix}\n")
            print(formatted)

            # Rich dashboard: signal panel (only when notable)
            readiness, signals, stressors = companion.get_signal_state()
            panel = brand.render_signal_panel(readiness, signals, stressors)
            if panel:
                print()
                print(panel)

            print()

    companion.end_session()
    print(brand.render_session_end())


def run_demo(companion: AuraCompanion) -> None:
    """Run the Maya career decision demo scenario from PRD v2.2 §19."""
    brand = get_brand()
    companion.start_session()

    demo_messages = [
        "I've been really stressed about the career decision. Can't stop thinking about whether to go independent.",
        "I didn't sleep well last night. The anxiety about leaving my job keeps me up.",
        "I saw a good setup on EUR/USD and took the trade anyway even though Buddy said no. Just felt like I needed a win.",
        "The trade went against me. Lost 40 pips. I knew Buddy was right but I overrode it.",
        "Maybe I should step back from trading when I'm this stressed about the career thing.",
    ]

    brand.print_header("Demo: Maya Career Decision Scenario", style="heavy")
    brand.print_dim("From PRD v2.2 §19 — demonstrating the feedback bridge")
    print()

    for i, msg in enumerate(demo_messages, 1):
        brand.print_divider("dots")
        brand.print_dim(f"Message {i}/{len(demo_messages)}")
        print(f"  {brand.c('success')}{brand.theme.fg('success')}Maya:{RESET} {msg}")
        print()

        response = companion.process_input(msg)
        prefix = brand.render_aura_prefix()
        print(f"  {prefix}{response}")
        print()

        # Show readiness after each message
        readiness_response = companion.process_input("/readiness")
        brand.print_dim(readiness_response)
        print()

    # Show final bridge and graph status
    brand.print_header("Final Status", style="heavy")
    print(companion.process_input("/bridge"))
    print()
    print(companion.process_input("/graph"))

    companion.end_session()
    print(brand.render_session_end())


def main():
    parser = argparse.ArgumentParser(
        prog="aura",
        description="Aura — Human Intelligence Engine",
    )
    parser.add_argument("--demo", action="store_true", help="Run the Maya demo scenario")
    parser.add_argument("--status", action="store_true", help="Show bridge status and exit")
    parser.add_argument("--theme", type=str, default=None,
                        choices=list(THEMES.keys()),
                        help="Set color theme (eve, midnight, aurora, ember, sakura, void, solaris, claude)")
    parser.add_argument("--db-path", type=str, default=None, help="Path to self-model database")
    parser.add_argument("--bridge-dir", type=str, default=None, help="Path to bridge signal directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging — keep the terminal clean unless --verbose
    if args.verbose:
        # Verbose: show everything on stderr
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        # Normal mode: log to file, keep terminal pristine
        log_dir = Path(".aura")
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            filename=str(log_dir / "session.log"),
            filemode="a",
        )

    # Initialize brand system
    brand = get_brand(theme_name=args.theme)

    # Apply theme from CLI arg
    if args.theme:
        brand.set_theme(args.theme)

    db_path = Path(args.db_path) if args.db_path else None
    bridge_dir = Path(args.bridge_dir) if args.bridge_dir else None

    companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)

    if args.status:
        # Fix M-02: Previously no try/except — a corrupted bridge file caused an unhandled crash.
        import json
        try:
            status = companion.bridge.get_bridge_status()
        except Exception as e:
            status = {"error": f"Bridge status unavailable: {e}", "degraded": True}
        print(json.dumps(status, indent=2, default=str))
        return

    # Print startup banner
    brand.print_startup_banner()
    print()

    if args.demo:
        run_demo(companion)
    else:
        run_interactive(companion)


if __name__ == "__main__":
    main()
