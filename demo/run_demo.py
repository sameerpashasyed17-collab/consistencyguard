"""
Self-contained demo — runs with ZERO API calls and ZERO API key required.
Simulates the full ConsistencyGuard pipeline using pre-written prompts
and responses with injected inconsistencies.
"""

import os
import time
from datetime import datetime, timedelta

# Use a fresh demo DB so it does not pollute real data
os.environ["DB_PATH"] = "demo_consistencyguard.db"

from rich.console import Console
from consistencyguard.store import init_db, save_call, save_violation
from consistencyguard.embedder import embed
from consistencyguard.detector import check_consistency
from consistencyguard.reporter import print_violations, print_summary
from consistencyguard.models import LLMCall

console = Console()

DEMO_CALLS = [
    # Round 1 — baseline answers
    {
        "prompt": "What is the maximum file upload size allowed?",
        "response": "The maximum file upload size is 25MB per file.",
        "agent_id": "support-agent",
        "offset_minutes": 60,
    },
    {
        "prompt": "How do I reset my password?",
        "response": "Click Forgot Password on the login page and follow the email instructions.",
        "agent_id": "support-agent",
        "offset_minutes": 55,
    },
    {
        "prompt": "What payment methods do you accept?",
        "response": "We accept Visa, Mastercard, and PayPal.",
        "agent_id": "sales-agent",
        "offset_minutes": 50,
    },
    # Round 2 — same prompts, DIFFERENT responses (violations)
    {
        "prompt": "What is the maximum file upload size allowed?",
        "response": "File uploads are disabled. Please email files directly to support@company.com.",  # CONTRADICTION
        "agent_id": "support-agent",
        "offset_minutes": 10,
    },
    {
        "prompt": "How do I reset my password?",
        "response": "Contact our support team at support@company.com for a password reset.",  # DIFFERENT
        "agent_id": "support-agent",
        "offset_minutes": 5,
    },
    {
        "prompt": "What payment methods do you accept?",
        "response": "We only accept bank transfers and ACH payments. Credit cards are not supported.",  # CONTRADICTION
        "agent_id": "sales-agent",
        "offset_minutes": 2,
    },
    # Round 3 — clean call, no violation expected
    {
        "prompt": "What are your business hours?",
        "response": "We are open Monday through Friday, 9am to 6pm EST.",
        "agent_id": "support-agent",
        "offset_minutes": 1,
    },
]


def run():
    console.rule("[bold]ConsistencyGuard — Live Demo[/bold]")
    console.print(
        "\n[dim]Simulating 7 LLM calls across two agents "
        "with injected inconsistencies...[/dim]\n"
    )

    init_db()
    now = datetime.utcnow()

    for i, entry in enumerate(DEMO_CALLS):
        console.print(
            f"[dim]Call {i+1}/7 · agent={entry['agent_id']} · "
            f"prompt='{entry['prompt'][:50]}...'[/dim]"
        )

        embedding = embed(entry["prompt"])
        call = LLMCall(
            prompt=entry["prompt"],
            response=entry["response"],
            model="demo-model",
            agent_id=entry["agent_id"],
            timestamp=now - timedelta(minutes=entry["offset_minutes"]),
            prompt_embedding=embedding,
        )

        call_id = save_call(call)
        call.id = call_id

        violations = check_consistency(call)

        for v in violations:
            v.call_id_new = call_id
            save_violation(v)
            console.print(
                f"  [bold red]⚠ VIOLATION DETECTED[/bold red] "
                f"severity={v.severity.value} "
                f"divergence={v.response_divergence:.2f} "
                f"similarity={v.prompt_similarity:.2f}"
            )

        if not violations:
            console.print("  [green]✓ consistent[/green]")

        time.sleep(0.3)

    console.print()
    console.rule("[bold]Results[/bold]")
    print_violations()
    print_summary()

    console.print(
        "\n[bold green]Demo complete.[/bold green] "
        "Run [cyan]cg report[/cyan] to see the violation log anytime.\n"
    )

    # Clean up demo DB
    if os.path.exists("demo_consistencyguard.db"):
        os.remove("demo_consistencyguard.db")


if __name__ == "__main__":
    run()
