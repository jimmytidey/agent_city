# agents/city_builder_agent.py
import json
from openai import AsyncOpenAI
from utils import update_msg

class CityBuilderAgent:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.system_prompt = """
            Imagine a grid-based city with a river, forest, grassland, and houses.
            Generate a compact city map.

            Legend (numeric codes):
            0 = grass
            1 = forest
            2 = river
            3 = house
            
            There should be a roughly equal proportion of grassland and forest.
            The river should be continuous and natural-looking, at least two cells wide, and must connect to at least one edge of the map.

            Return STRICT JSON ONLY (no code fences, no prose):
            { "grid": [[row0], [row1], ..., [row9]] }

            Rules:
            - Grid size exactly 10 rows × 20 columns.
            - Each row is a list of 20 integers using only the codes 0,1,2,3.
            - Houses must be clustered together (not scattered) — about ~10 houses total.
            - River must form a continuous course — at least ~15 river cells.
            - Grassland and forest should appear in roughly equal proportions.

            Example: 
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 2, 2], [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 2, 2, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            """

    async def build_city_json(self) -> dict:

        await update_msg("⏳ Building map")

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Return the JSON now."},
                ],
                temperature=1,
                response_format={"type": "json_object"},  # force JSON
            )
            txt = resp.choices[0].message.content.strip()
            data = json.loads(txt)
           
            grid = data.get("grid")
            
            if not isinstance(grid, list) or len(grid) != 10:
                raise ValueError("grid must be a list of 10 strings")

            return data
        except Exception as e:
            return {"error": f"⚠️ Parse error: {e}"}
