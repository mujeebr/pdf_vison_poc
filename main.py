# import streamlit as st
# conda install poppler 
# pdfvision env
from typing_extensions import TypedDict
from pdf2image import convert_from_path
from io import BytesIO
import openai
import base64
# from core.gpt_connection import llm_model
from langgraph.graph import START, END, StateGraph


# Define the state,
class State(TypedDict):
    file_path: str
    base64_images: list
    extracted_info: str
    processed_info: str

# Nodes
def load_pdf(state: State):
    """Load the PDF and convert to base64 images"""
    POPPLER_PATH = "/Users/shaikmujeeburrahman/opt/anaconda3/envs/pdfvision/bin"

    images = convert_from_path(state['file_path'], poppler_path=POPPLER_PATH)
    # images = convert_from_path(state['file_path']

    base64_images = []

    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="JPEG") # buffered will hold the binary data of the image in memory
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(base64_str)
        # buffered.getvalue() extracts the binary contents from BytesIO.
        # base64.b64encode(...) converts this binary data into Base64 encoding, which allows us to represent binary files as text (useful for embedding images in HTML, JSON, etc.).
        # .decode("utf-8") converts the Base64 bytes into a string, making it easy to store or send.

    state['base64_images'] = base64_images
    return state


def extract_information(state: State):
    """Extract information from the base64 images"""
    question = """what is this document about?"""

    # Prepare the messages payload
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]

    # Attach images to the message
    for img in state['base64_images']:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    # messages = [{"role": "user", "content": question}]
    #
    # for img in state['base64_images']:
    #     messages.append({
    #         "role": "user",
    #         "content": f"data:image/jpeg;base64,{img}"
    #     })

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    state['extracted_info'] = response.choices[0].message.content

    # response = llm_model.invoke(input=messages)
    # state['extracted_info'] = response.content
    return state


def process_information(state: State):
    """Process the extracted information"""
    state['processed_info'] = state['extracted_info']
    return state


# Build the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("load_pdf", load_pdf)
workflow.add_node("extract_information", extract_information)
workflow.add_node("process_information", process_information)

# Add edges to connect nodes
workflow.add_edge(START, "load_pdf")
workflow.add_edge("load_pdf", "extract_information")
workflow.add_edge("extract_information", "process_information")
workflow.add_edge("process_information", END)

app = workflow.compile()


# Streamlit app
# def main():
#     st.title("PDF Signature Extractor")

#     uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
#     img_path = app.get_graph().draw_mermaid_png()

#     # Display the image using Streamlit
#     st.image(img_path)
#     if uploaded_file is not None:
#         with open("temp.pdf", "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Initialize the state
#         state = {'file_path': "temp.pdf"}

#         # Compile and run the workflow
#         chain = workflow.compile()
#         final_state = chain.invoke(state)

#         st.subheader("Extracted Information")
#         st.text(final_state['processed_info'])


# Initialize the state
pdf_path=r"/Users/shaikmujeeburrahman/Downloads/Mujeeb_AI_CV-3.pdf"

state = {'file_path': pdf_path}

# Compile and run the workflow
chain = workflow.compile()
final_state = chain.invoke(state)
print(final_state['processed_info'])

# if __name__ == "__main__":
#     main()
