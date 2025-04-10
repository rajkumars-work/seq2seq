# coglib

Library and command-line tools for the Cogs (Contextual Generator Seq-to-seq system 

This is the techdocs documentation for coglib.
## Developing the command-line interface

This repo comes pre-configured with [Typer] (built on the also-excellent [click])
for building CLIs. You just need to declare commands on the Typer `app` (as
shown in `src/coglib/cli.py`) and use Python's type
annotation syntax to declare command-line inputs - no arg-parsing, piles of decorators,
or manual formatting. For example,

```python
app = typer.Typer()

@app.command("hello")
def hello(
    name: str = typer.Option(
        ..., "--name", prompt="Your name", help="The person to say hello to."
    ),
    count: int = typer.Option(1, "-c", "--count", help="Number of greetings."),
    capitalize: bool = typer.Option(False, help="Force all caps."),
) -> None:
    """Greet a user.

    This demonstrates using Typer CLI options, including using prompts for user
    input, short/long flag names, and auto-generating flags for boolean options.
    """
    ...
```

generates a CLI command like

```sh
$ coglib hello --help

 Usage: coglib hello [OPTIONS]

 Greet a user.
 This demonstrates using Typer CLI options, including using prompts for user input, short/long flag names, and auto-generating flags for boolean options.

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --name                               TEXT     The person to say hello to. [default: None] [required]                    │
│    --count       -c                     INTEGER  Number of greetings. [default: 1]                                         │
│    --capitalize      --no-capitalize             Force all caps. [default: no-capitalize]                                  │
│    --help                                        Show this message and exit.                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Which can be accessed via the command defined as an entrypoint script in `pyproject.toml`.

If this looks like a [FastAPI] app, you're not wrong - Typer was built by the same
developer, and follows many of the same design patterns.

<!-- Sources -->
[Typer]: https://typer.tiangolo.com/
[click]: https://click.palletsprojects.com/en/8.1.x/
[FastAPI]: https://fastapi.tiangolo.com/
