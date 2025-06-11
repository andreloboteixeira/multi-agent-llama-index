import asyncio
import os
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.llms.openai import OpenAI


async def read_notes(ctx: Context):
    """Read notes from a markdown file asynchronously"""

    def _read():
        try:
            with open("src/lobo/notes.md", "r") as file:
                return file.read()
        except FileNotFoundError:
            return "No notes file found."

    return await asyncio.to_thread(_read)


async def write_refined_todos(ctx: Context, todos: str):
    """Write refined todos to a markdown file asynchronously"""

    def _write():
        with open("src/lobo/refined_todos.md", "w") as file:
            file.write(todos)

    await asyncio.to_thread(_write)


def build_notes_organizer_agent(llm: OpenAI):
    """Build a notes organizer agent"""
    return FunctionAgent(
        name="NotesOrganizerAgent",
        description="Organize notes into refined todos",
        system_prompt="You are a helpful assistant that reads notes "
        "and writes organized todos.",
        llm=llm,
        tools=[read_notes, write_refined_todos],
        can_handoff_to=["PrioritizationAgent"],
    )


def build_prioritization_agent(llm: OpenAI):
    """Build a prioritization agent"""
    return FunctionAgent(
        name="PrioritizationAgent",
        description="Prioritize todos",
        system_prompt="You are a helpful assistant that prioritizes todos.",
        llm=llm,
        tools=[],
        can_handoff_to=["NotesOrganizerAgent"],
    )


async def main():
    """Main function to run the multi-agent workflow"""
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    llm = OpenAI(model="gpt-4o-mini")

    notes_organizer_agent = build_notes_organizer_agent(llm)
    prioritization_agent = build_prioritization_agent(llm)

    workflow = AgentWorkflow(
        agents=[notes_organizer_agent, prioritization_agent],
        root_agent=notes_organizer_agent.name,
        initial_state={
            "notes": "",
            "todos": ""
        },
    )
   
    handler = workflow.run(
        user_msg="""
            Organize my notes into refined todos.
            """
    )

    # Print workflow events to follow execution progress
    current_agent = None
    async for event in handler.stream_events():
        if (hasattr(event, 'current_agent_name') and
                event.current_agent_name != current_agent):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")
        elif isinstance(event, AgentOutput):
            if event.response and hasattr(event.response, 'content'):
                print("📤 Output:", event.response.content)
            if hasattr(event, 'tool_calls') and event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
        else:
            print(f"📋 Event: {type(event).__name__}")
            if hasattr(event, 'result'):
                print(f"  Result: {event.result}")


if __name__ == "__main__":
    asyncio.run(main())
