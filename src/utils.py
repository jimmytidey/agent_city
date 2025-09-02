from typing import List, Dict
import chainlit as cl


def numbers_to_emojis(grid: list[list[int]]) -> str:
    """Convert a numeric grid into a single emoji string with newlines."""
    mapping = {
        0: "🟩",  # grass
        1: "🌳",  # forest
        2: "🟦",  # river
        3: "🏠",  # house
        10: "🟥"  # new houses (red overlay)
    }
    rows = ["".join(mapping.get(cell, "❓") for cell in row) for row in grid]
    return "\n".join(rows)

def apply_cells_as_new_houses(grid: List[List[int]], cells: List[List[int]]) -> List[List[int]]:
    new_grid = [row[:] for row in grid]
    for r, c in cells:
        new_grid[r][c] = 10  # new houses
    return new_grid

async def update_msg(msg): 
    reasoning_spinner = cl.Message(content=msg)
    await reasoning_spinner.send()

def append_to_history(new_entry: str): 
    previous_history = cl.user_session.get("history", "")
    updated_history = f"{previous_history}\n\n{new_entry}"
    cl.user_session.set("history", updated_history)