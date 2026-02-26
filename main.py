from langgraph.graph import StateGraph, END
from graph_nodes import AgentState, input_loader_node, extract_audio_node, transcribe_node, visual_description_node, notes_generator_node, vision_router

def create_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("input", input_loader_node)
    workflow.set_entry_point("input")

    workflow.add_node("audio",extract_audio_node)
    workflow.add_edge("input","audio")

    workflow.add_node("caption",transcribe_node)
    workflow.add_edge("audio","caption")

    workflow.add_conditional_edges("caption", vision_router,{"visual":"visual","note":"notes"})

    workflow.add_node("visual",visual_description_node)

    workflow.add_node("notes",notes_generator_node)
    workflow.add_edge("visual","notes")
    workflow.add_edge("notes",END)

    return workflow.compile()

app = create_workflow()