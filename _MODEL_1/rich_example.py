from rich.console import Console
from rich.table import Table

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("ID", style="dim", width=10)
table.add_column("Description")
table.add_column("Quantity")

table.add_row("123456", "Widget", "10")
table.add_row("234567", "Gadget", "20")
table.add_row("345678", "Thingy", "30")

console.print(table)

# ------------------------------------------------

from rich.syntax import Syntax

console = Console()

code = """
def hello_world():
    print("Hello, world!")
"""

syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
console.print(syntax)

# -------------------------------------------------

from rich.progress import track
for i in track(range(1000000)):
    a = i + 1

# -------------------------------------------------

from rich.markdown import Markdown

console = Console()

markdown = Markdown("# Hello, world!")

console.print(markdown)

# -------------------------------------------------

from rich import print

print(":thumbs_up: Good job!")


# Named Colors: Rich supports the standard set of CSS color names. Some examples include:
# "red"
# "green"
# "blue"
# "cyan"
# "magenta"
# "yellow"
# "white"
# "black"

# Hex Colors: You can use hexadecimal color codes:
# "#FF0000" for red
# "#00FF00" for green
# "#0000FF" for blue

# RGB Colors: You can specify RGB colors using tuples:
# (255, 0, 0) for red
# (0, 255, 0) for green
# (0, 0, 255) for blue
 
# Example:
# "bold red on white"