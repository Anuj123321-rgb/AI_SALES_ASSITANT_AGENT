from src.agent import AutoStreamAgent


def run_cli() -> None:
    agent = AutoStreamAgent()
    state = None

    print("AutoStream Agent is ready. Type 'exit' to quit.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Agent: Goodbye!")
            break

        state = agent.chat(user, state)
        print(f"Agent: {state['messages'][-1].content}")


if __name__ == "__main__":
    run_cli()
