from graph_nodes import (
    input_loader_node, 
    extract_audio_node, 
    transcribe_node, 
    visual_description_node,
    notes_generator_node
)

state = {"user_input": "https://www.youtube.com/watch?v=FWI9GEwJNzc"}

print("Running Final Pipeline...")
state.update(input_loader_node(state))
state.update(extract_audio_node(state))
state.update(transcribe_node(state)) 
state.update(visual_description_node(state))
state.update(notes_generator_node(state))

print("\n" + "="*50)
print("FINAL AI GENERATED NOTES")
print("="*50)
print(state["final_notes"])