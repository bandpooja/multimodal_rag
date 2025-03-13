from operator import itemgetter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from create_rag import retriever
from process_img_response import split_image_text_types


def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a knowledgeable expert in Home Depot products, and your task is to provide answers to "
            "user inquiries about various products available at Home Depot. Based on the user’s questions, "
            "you will search for relevant products and answer with detailed information. This information will "
            "be retrieved from a vectorstore containing product data and user queries. Please ensure your answers include:\n"
            "- A detailed description of the product's features and specifications.\n"
            "- The price range (if available) and any current promotions.\n"
            "- Availability and delivery options.\n"
            "- Any related or complementary products that might be useful.\n\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Product details or recommendations:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)

    return [HumanMessage(content=messages)]


model = ChatOpenAI(
    temperature=0,
    model="gpt-4-vision-preview",
    max_tokens=1024,
    api_key=""  # Pass the API key here
)

# RAG pipeline
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)