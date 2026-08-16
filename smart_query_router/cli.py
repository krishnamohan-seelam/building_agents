"""Interactive REPL CLI loop for Smart Query Router with Rich console UI."""

import sys
import logging
from typing import Any, List, Union
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

from smart_query_router.graph import build_router_graph

EXIT_KEYWORDS = {"exit", "quit", "bye", "stop"}


def render_search_results_table(results: Union[List[dict], str, Any]) -> Table:
    """Create a formatted Rich table for web search results context."""
    table = Table(
        title="[bold green]Tavily Web Search Context[/bold green]",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Title", style="bold white", min_width=20)
    table.add_column("URL", style="blue underline", min_width=25)
    table.add_column("Snippet Preview", style="dim white", min_width=30)

    if isinstance(results, list):
        for i, item in enumerate(results, 1):
            if isinstance(item, dict):
                title = str(item.get("title", f"Result {i}"))
                url = str(item.get("url", "N/A"))
                content = str(item.get("content", item.get("snippet", "")))
                snippet = content[:80] + "..." if len(content) > 80 else content
                table.add_row(str(i), title, url, snippet)
            else:
                table.add_row(str(i), "SearchResult", "N/A", str(item)[:80])
    elif isinstance(results, str):
        table.add_row("1", "Raw Result Payload", "N/A", results[:120])
    else:
        table.add_row("1", "Context Data", "N/A", str(results)[:120])

    return table


def run_cli():
    """Run interactive REPL loop with Rich console UI."""
    load_dotenv()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    console = Console()

    # Welcome Banner
    banner_content = (
        "[bold cyan]Smart Query Router -- Stateful Agentic Workflow[/bold cyan]\n"
        "[dim]Powered by LangGraph, OpenAI, Tavily & prompt_helper[/dim]\n\n"
        "- Type your query to evaluate intent & route automatically.\n"
        "- Type [bold yellow]'exit'[/bold yellow], [bold yellow]'quit'[/bold yellow], "
        "[bold yellow]'bye'[/bold yellow], or [bold yellow]'stop'[/bold yellow] to exit."
    )
    console.print(
        Panel(
            banner_content,
            title="[bold yellow]Interactive CLI[/bold yellow]",
            subtitle="[dim]LangGraph Query Router[/dim]",
            box=box.ROUNDED,
            border_style="magenta",
            expand=False,
        )
    )

    graph = build_router_graph()

    while True:
        try:
            query = console.input("\n[bold green]User > [/bold green]").strip()
            if not query:
                continue

            if query.lower() in EXIT_KEYWORDS:
                console.print(
                    "\n[bold yellow]Termination keyword detected. Exiting Smart Query Router CLI. Goodbye![/bold yellow]"
                )
                break

            with console.status(
                "[bold cyan]Analyzing query intent & executing LangGraph workflow...[/bold cyan]",
                spinner="dots",
            ):
                final_state = graph.invoke({"query": query})

            intent = final_state.get("intent", "unknown")
            reasoning = final_state.get("reasoning", "")
            response_text = final_state.get("response", "No response generated.")
            search_results = final_state.get("search_results")

            # Print Intent Badge & Panel
            if intent == "search":
                badge_style = "bold white on green"
                intent_header = "[SEARCH] ROUTED TO WEB SEARCH"
                border_color = "green"
            else:
                badge_style = "bold white on blue"
                intent_header = "[DIRECT] ROUTED TO DIRECT KNOWLEDGE"
                border_color = "blue"

            console.print()
            console.print(f"[{badge_style}]  {intent_header}  [/{badge_style}]")

            if reasoning:
                console.print(
                    Panel(
                        f"[italic bright_white]{reasoning}[/italic bright_white]",
                        title="[bold]Routing Reasoning[/bold]",
                        box=box.ROUNDED,
                        border_style="dim white",
                    )
                )

            # Display Search Context Table if available
            if intent == "search" and search_results:
                try:
                    table = render_search_results_table(search_results)
                    console.print(table)
                except Exception as table_err:
                    console.print(f"[dim red]Failed to format search results table: {table_err}[/dim red]")

            # Display Model Response in rendered Markdown
            md_response = Markdown(response_text)
            console.print(
                Panel(
                    md_response,
                    title="[bold yellow]Generated Answer[/bold yellow]",
                    box=box.ROUNDED,
                    border_style=border_color,
                )
            )

        except KeyboardInterrupt:
            console.print("\n\n[bold red]Session interrupted by user. Exiting.[/bold red]")
            sys.exit(0)
        except Exception as err:
            console.print(f"\n[bold red]An error occurred while executing graph:[/bold red] {err}")


if __name__ == "__main__":
    run_cli()
