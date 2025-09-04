# agents/developer_agent.py
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import json
import chainlit as cl
from utils import numbers_to_emojis, apply_cells_as_new_houses, update_msg

class RenderProposalAgent:
    def __init__(self, model: str = "gpt-5", client: Optional[AsyncOpenAI] = None):
        self.model = model
        self.client = client or AsyncOpenAI()

    async def render_proposal(self, grid: List[List[int]], prosposal_as_text: Optional[str] = None) -> Dict[str, Any]:
        """
        rows: 10x20 grid of ints {0,1,2,3}
        context: str (optional) - reasoning from the reasoning agent
        returns: {"cells": [[r,c], ... 8 total], "justification": "..."}
        """

        # Ensure update_msg is awaited
        await update_msg("⏳ Adding proposal to map.")

        system_msg = {
            "role": "system",
            "content": (
                "You are an assistant that generates housing proposals on a 10x20 grid."
                "You are given the textual reasoning from a resident or developer agent about where to place new houses as the variable: prosposal_as_text"
                "You must strictly return JSON with the following structure:\n\n"
                "{\n"
                "  \"cells\": [[row, col], [row, col], ...],\n"
                "}\n\n"
                "Rules:\n"
                "- The 'cells' key must contain exactly 8 unique [row, col] pairs.\n"
                "- Each [row, col] pair must be within the grid bounds.\n"
                "- coordinates are zero indexed"
                "- The 'justification' key must explain why these cells were chosen.\n"
                "Do not include any additional text outside the JSON structure."
                "Example: \n"
                "Given grid:    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 2, 2], [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 2, 2, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] \n \n "
                "And proposal_as_text: The best place for houses is close to the existing houses on the bottom two rows on the right, aligned with the water. \n"
                "You might reply with: {\"cells\": [[8, 15], [8, 16], [8, 17], [8, 18], [9, 15], [9, 16], [9, 17], [9, 18]]}"
            )
        }

        user_msg = {
            "role": "user",
            "content": json.dumps({
                "task": "Propose 8 cells for new houses on the map. NEVER place houses on river or existing houses.",
                "prosposal_as_text": prosposal_as_text,
                "legend": {"0": "grass", "1": "forest", "2": "river", "3": "existing house", "10": "new house"},
                "grid_shape": [len(grid), len(grid[0]) if grid else 0],
                "grid": grid,
                "constraints": {
                    "cells_required": 8,
                    "place_on": 0,
                    "avoid": [1, 2, 3],                    
                    "clustered": True,
                    "avoid_river_cells": True,   # Do not overwrite river cells
                    "avoid_existing_houses": True # Do not overwrite existing houses
                },
                
            })
        }

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[system_msg, user_msg],
            temperature=0.2,
            response_format={"type": "json_object"}  # Force JSON response
        )

        proposal_json = resp.choices[0].message.content

        # Defensive parsing
        try:
            print("Raw proposal_json:", proposal_json)  # Debugging
            data = json.loads(proposal_json)
            cells = data.get("cells", [])
            justification = data.get("justification", "").strip()

            # Basic validation: exactly 8 cells, each [r, c] in-bounds
            if len(cells) != 8:
                raise ValueError("Expected exactly 8 cells.")
            R, C = len(grid), len(grid[0]) if grid else 0
            for rc in cells:
                if not (isinstance(rc, list) and len(rc) == 2):
                    raise ValueError("Each cell must be [row, col].")
                r, c = rc
                if not (0 <= r < R and 0 <= c < C):
                    raise ValueError(f"Cell out of bounds: {rc}")

            proposal_grid = apply_cells_as_new_houses(grid, cells)
            print("proposal_grid:", proposal_grid)  # Debugging
            emoji_map = numbers_to_emojis(proposal_grid)
            print("emoji_map:", emoji_map)  # Debugging
            await cl.Message(author="Proposal_render_agent", content=emoji_map).send()

            return proposal_grid

        except Exception as e:
            print(f"Error parsing proposal: {e}")
            # Fallback: return empty result with error message (so caller can handle gracefully)
            return {"cells": [], "justification": f"⚠️ Failed to parse proposal: {e}"}