from main import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_q_function_definition():
    assert validate(
        question="what is a Q-Function",
        expected_response="a function that estimates the expected cumulative reward for taking a specific action in a given state and then following a certain policy",
    )


def test_policy_definition():
    assert validate(
        question="what is a policy and what types can it be",
        expected_response="A mapping from states to actions that defines the agent’s behavior. A policy can be deterministic (a fixed action for each state) or stochastic (assigning probabilities to actions in a given state)"
    )


def validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    
    if "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False

    raise ValueError(
        f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
    )
