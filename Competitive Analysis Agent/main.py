import os
from dotenv import load_dotenv
from agent.data_loader import load_and_prepare_data
from agent.agent import CompetitiveAnalysisAgent

def main():
    load_dotenv()
    cohere_key = os.getenv("COHERE_API_KEY")

    print("Loading competitor dataset...")
    df = load_and_prepare_data("data/competitors.csv")

    agent = CompetitiveAnalysisAgent(df, cohere_key)

    print("\n=== AI Competitive Analysis Agent ===")
    print("Type your queries, or use:")
    print("  history → view past queries")
    print("  logs → view reasoning steps")
    print("  exit → quit\n")

    while True:
        user_query = input("\nEnter Query: ")

        if user_query == "exit":
            break

        elif user_query == "history":
            print("\n--- Query History ---")
            for h in agent.show_history():
                print(f"Q: {h['query']}\nA: {h['response']}\n")
            continue

        elif user_query == "logs":
            print("\n--- Reasoning Logs ---")
            for log in agent.show_reasoning_logs():
                print(json.dumps(log, indent=2))
            continue

        else:
            answer, reasoning = agent.reason_and_act(user_query)
            print("\nANSWER:\n", answer)

            print("\nREASONING STEPS:")
            for step in reasoning:
                print(" -", step)


if __name__ == "__main__":
    main()
