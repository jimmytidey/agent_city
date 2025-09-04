# agents/developer_agent_reasoning.py
import os, json
from typing import List
from openai import AsyncOpenAI
from utils import update_msg
import chainlit as cl

class DeveloperReasoningAgent:
    """
    LLM agent that explains WHERE an 8-cell (contiguous) housing block should go,
    in plain language. It does NOT return coordinates or JSON — just reasoning.
    """

    def __init__(self, model: str = "gpt-5", temperature: float = 1.0):
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        self.system = (
            "You are a planning-savvy developer agent. You receive a 2D grid map with values:\n"
            "0=grass, 1=forest, 2=river, 3=house.\n"
            "Your task: verbally explain the best place for a contiguous 8-cell housing block "
            "On grass, close to existing houses, and with views of rivers and forests.\n"
            "Output REQUIREMENTS:\n"
            "- Reply in plain language only (no JSON, no coordinates, no code fences).\n"
            "- Use concise bullet points, then a 1–2 sentence recommendation.\n"
            "- You cannot place houses in forested areas - you are not allowed to build there.\n"
            "- You cannot place houses on river cells - you are not allowed to build there.\n"
            "- You should prefer cells close to water - buyers prefer views of water.\n"
            "- You should try to build near existing housing - buyers want good accces.\n"
            "- Do not mention the number system.\n"
            "- Make explicit reference to previous resident proposals provided in the negotiation history and proposal_grid.\n"
            "- If the residents have made a counter proposal at least once in the negotiation_history, and if the residents proposal is sufficently near houses and water, reply with {TERMINATE NEGOITATION]. \n"   
            "- Always be explicit about your intended location of the new houses. Do not just say eg. 'it should stay where it was previously'. \n"            
            "- Answer as though you are a housing developer. If previous proposals are provided in the negotiation history, bargain with the residents to get the best deal you can.\n"
            "Example: \n "
            "Given grid:    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 2, 2], [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 2, 2, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] \n" 
            "You might say: The best place for houses is close to the existing houses on the bottom two rows on the right, aligned with the water. This will allow for river views and good access. \n"
        )

    async def reason(self, grid: List[List[int]], negotiation_history:str, first_turn:bool):

        await update_msg("⏳ Developer Agent considering locations")

        if first_turn:
            negotiation_history = "There have not been any previous proposals" 

        user_payload = {
            "legend": {"0": "grass", "1": "forest", "2": "river", "3": "house", '10': 'proposed new house'},
            "grid_shape": [len(grid), len(grid[0]) if grid else 0],
            "grid": grid,
            "negotiation_history": negotiation_history,
            "proposal_grid": grid,
            "constraints": {
                "place_on": 0,
                "avoid": [1, 2, 3],
                "prefer_near_houses": True,
                "prefer_near_rivers": True
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

        developer_reasoning = resp.choices[0].message.content.strip()

        await cl.Message(
            author="Developer Agent",
            content=developer_reasoning
        ).send()

        return developer_reasoning
