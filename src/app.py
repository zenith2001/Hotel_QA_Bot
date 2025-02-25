from generate_answer import generate_answer

def main():
    """Interactive CLI for the Hotel Policy QA Bot."""
    print("\n🏨 Hotel Policy QA Bot - Ask Your Questions!")
    print("🔹 Type 'exit' to quit.\n")

    hotel_name = input("Enter the hotel name: ").strip()

    while True:
        user_query = input("\nYou: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        # Generate and display the answer
        answer = generate_answer(user_query, hotel_name)
        print(f"\n🤖 Bot: {answer}\n")

if __name__ == "__main__":
    main()
