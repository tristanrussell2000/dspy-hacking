import dspy
import os
from dotenv import load_dotenv

load_dotenv()
lm = dspy.LM(model="gemini/gemini-2.5-flash", max_tokens=65536)
dspy.settings.configure(lm=dspy.LM("gemini/gemini-2.5-flash"))

class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
history = dspy.History(messages=[])

def main():
    while True:
        question = input("Question: ")
        if question == "end":
            break
        outputs = predict(question=question, history=history)
        print(f"\n{outputs.answer}\n")
        history.messages.append({"question": question, **outputs})

    dspy.inspect_history()

if __name__ == "__main__":
    main()
