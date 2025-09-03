# agents/developer_agent_reasoning.py
import os, json
from typing import List
from openai import AsyncOpenAI
import chainlit as cl

class ResidentReasoningAgent:
    """
    LLM agent that explains WHERE an 8-cell (contiguous) housing block should go,
    in plain language. It does NOT return coordinates or JSON — just reasoning.
    """

    def __init__(self, model: str = "gpt-5", temperature: float = 1):
        self.model = model
        self.temperature = temperature
        # Uses OPENAI_API_KEY from env
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        self.system = (
            "You are a resident of an imaginary village. You are provided with a 2D grid map with of the village with values:\n"
            "0=grass, 1=forest, 2=river, 3=house, 10=proposed new house.\n"
            "Your task: verbally explain the best place for a contiguous 8-cell housing block in your village.\n"
            "You should try to position the proposed houses away from existing houses.\n"
            "Please take into account the previous proposals provided in the negotiation history.\n"
            "Output REQUIREMENTS:\n"
            "- Reply in plain language only (no JSON, no coordinates, no code fences).\n"
            "- Concise bullet points, then a 1–2 sentence recommendation.\n"
            "- Do not mention the numbering scheme.\n"
            "- Reply as though you are concerned resident, protecting forest, rivers and, most of all, existing houses.\n"
            "- Make explicit reference to previous developer proposals in the negotiation history and.\n"
            "- If previous proposals are provided in the negotiation history, bargain with the developer to get the new, proposed houses as far away from te existing houses as possible.\n"
            "- If previous proposals seem acceptable, agree to them. State that you agree and propose the same map.\n"   

        )

    async def reason(self, grid: List[List[int]], negotiation_history:str) -> str:
        """
        Returns a short verbal rationale (markdown bullets + short recommendation).
        """

        reasoning_spinner = cl.Message(content="⏳ Resident Agent considering locations.")
        await reasoning_spinner.send()

        user_payload = {
            "legend": {"0": "grass", "1": "forest", "2": "river", "3": "house", '10': 'proposed new house'},
            "grid_shape": [len(grid), len(grid[0]) if grid else 0],
            "proposed_grid": grid,
            "Negotiation_history": negotiation_history,
            "constraints": {
                "place_on": 0,
                "avoid": [1,2,3],
                "prefer_away_from_existing_houses": True,               
            }
        }

        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )

        resident_reasoning = resp.choices[0].message.content.strip()

        await cl.Message(
            author="Resident Agent (Reasoning)",
            content=resident_reasoning
        ).send()

        return resident_reasoning
