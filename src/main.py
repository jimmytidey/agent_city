import chainlit as cl
from openai import AsyncOpenAI
from utils import numbers_to_emojis, apply_cells_as_new_houses, update_msg, append_to_history

from agents.city_builder_agent import CityBuilderAgent
from agents.developer_reasoning_agent import DeveloperReasoningAgent  
from agents.resident_reasoning_agent import ResidentReasoningAgent
from agents.render_proposal_agent import RenderProposalAgent

client = AsyncOpenAI()
render_proposal = RenderProposalAgent(model="gpt-4o-mini")
city_builder_agent = CityBuilderAgent(client=client, model="gpt-4o-mini")
dev_reasoner = DeveloperReasoningAgent(model="gpt-4o-mini")
resident_reasoner = ResidentReasoningAgent(model="gpt-4o-mini")

@cl.on_chat_start
async def start():
    city: dict # will hold the city data including the grid
    grid: list[list[int]] # will hold the numeric grid representation of the city
    proposal_grid: dict = {} # will hold the proposed grid from render_proposal
    emoji_map: str # will hold the emoji representation of the map
    developer_reasoning: str # will hold the reasoning from the developer agent
    resident_reasoning: str # will hold the reasoning from the resident agent

    # 1) Build city
    city = await city_builder_agent.build_city_json()
    grid = city["grid"]
    emoji_map = numbers_to_emojis(grid)
    await cl.Message(author="city_builder_agent", content=emoji_map).send()
    append_to_history(f"Initial city map: {grid}")
    cl.user_session.set('proposal_grid', grid)
    cl.user_session.set('first_turn', True)

    # 2) iterative negotiation rounds
    for round_no in range(4):  

        # Developer
        ## Reasoning 
        developer_reasoning = await dev_reasoner.reason(
            grid = cl.user_session.get('proposal_grid'), 
            negotiation_history=cl.user_session.get("history"), 
            first_turn= cl.user_session.get('first_turn')
        ) 
        append_to_history(f"Developer reasoning round {round_no}: {developer_reasoning}")

        ## Propose grid  
        proposal_grid = await render_proposal.render_proposal(grid, context=developer_reasoning)
        cl.user_session.set('proposal_grid', proposal_grid)
        cl.user_session.set('first_turn', False)
        
        # Resident 
        ## Reasoning 
        print(cl.user_session.get('proposal_grid'))
        resident_reasoning = await resident_reasoner.reason(
            grid = cl.user_session.get('proposal_grid'), 
            negotiation_history=cl.user_session.get("history"), 
        )             
        append_to_history(f"Resident reasoning round {round_no}: {resident_reasoning}")

        ## propose grid 
        proposal_grid = await render_proposal.render_proposal(grid, context=resident_reasoning)
        append_to_history(f"Developer proposal grid round {round_no}: {proposal_grid}")
        cl.user_session.set('proposal_grid', proposal_grid)

