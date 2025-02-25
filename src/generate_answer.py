from retriever import retrieve_answer
from langchain_community.llms import LlamaCpp
from config import MODEL_PATH

# Load the Mistral-7B model with optimized parameters
llm = LlamaCpp(
    model_path=MODEL_PATH,  
    temperature=0.1,  # Reduce randomness for factual responses
    max_tokens=500,  # Prevent excessive verbosity
    n_ctx=4096,  
    top_p=0.7,  # Restrict probability space to avoid hallucination
    verbose=False  
)

def generate_answer(user_query, hotel_name, debug=False):
    """Retrieves relevant hotel policies and generates an LLM response."""
    
    # Retrieve relevant documents based on the selected hotel
    retrieved_docs = retrieve_answer(user_query, hotel_name)

    if not retrieved_docs or all(doc.strip() == "" for doc in retrieved_docs):
        return "Sorry, no relevant information was found for this hotel."

    # Deduplicate retrieved chunks
    unique_chunks = list(set(retrieved_docs))

    # Debugging: Print retrieved chunks
    if debug:
        print("\n🔹 **Retrieved Chunks for Context:**")
        for idx, doc in enumerate(unique_chunks, 1):
            print(f"Chunk {idx}: {doc[:300]}...\n{'-'*80}")  # Print only first 300 characters

    # Combine retrieved documents into a single formatted context
    context = "\n\n".join(f"- {doc.strip()}" for doc in unique_chunks)

    # Few-shot examples to improve response accuracy
    few_shot_examples = """
    **Example 1:**  
    **Q:** What is the cancellation policy if I cancel 20 days before arrival?  
    **A:** If you cancel 20 days before arrival, a 50%\ retention charge from the full amount will apply.

    **Example 2:**  
    **Q:** What happens if I cancel my booking just 5 days before arrival?  
    **A:** If you cancel within 7 days of arrival, a 100%\ retention charge from the full amount will apply.

    **Example 3:**  
    **Q:** Will I get a refund if I cancel my booking during peak season?  
    **A:** No, there is no refund for cancellations during peak period bookings.

    **Example 4:**  
    **Q:** What documents are required for check-in?  
    **A:** It is mandatory to produce a valid government-issued photo ID or a passport with a visa page at the time of check-in.

    **Example 5:**  
    **Q:** What happens if someone misbehaves in the hotel premises?  
    **A:** The hotel reserves the right to ask any person who is not properly attired or misbehaves to leave the hotel premises immediately.
    """

    # Create the prompt for the LLM
    prompt = f"""
    You are a professional hotel assistant providing policy-related answers.  
    **Your response must be strictly factual, concise, and based ONLY on the given policies.**  

    **Rules for Answering:**  
    - **Do not add extra details** beyond what is in the policy.  
    - **Do not include promotional content** or greetings.  
    - **If the answer is not in the retrieved policy, respond with 'I do not have this information.'**  

    **Hotel Name:** {hotel_name}  

    Below are correct response examples:  
    {few_shot_examples}

    **Retrieved Hotel Policy Information:**  
    ------------------  
    {context}  
    ------------------  

    **Now, answer the following question based only on the provided policy:**  
    **Customer's Question:** "{user_query}"  
    **Your Response:**  
    """
    print('Finding answer... Please wait...\n')
    # Generate the response using Mistral-7B
    response = llm.invoke(prompt)

    # Clean up response (removing unnecessary whitespace or artifacts)
    return response.strip()

# Run a test case
if __name__ == "__main__":
    query = "What is the cancellation policy?"
    hotel = "Cloudcastle_resort"
    print(generate_answer(query, hotel, debug=True))
