"""
File: answer_generator.py

Purpose:
Builds prompts and generates source-grounded financial answers from
retrieved news documents.

Role in Pipeline:
Generation Layer – Turns retrieved context into a structured response.
"""


class AnswerGenerator:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def build_context(self, documents: list[dict]) -> str:
        context_blocks = []

        for i, doc in enumerate(documents, start=1):
            block = f"""
            Source {i}
            Headline: {doc.get("headline")}
            Publisher: {doc.get("source")}
            URL: {doc.get("url")}
            Text: {doc.get("text")}
            """
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def build_prompt(self, question: str, documents: list[dict]) -> str:
        context = self.build_context(documents)

        return f"""
        You are a financial research assistant.

        Use only the retrieved context below to answer the user's question.
        Do not make investment recommendations.
        Do not tell the user to buy, sell, or hold a stock.
        If the context is not enough, say that the available sources are limited.

        User question:
        {question}

        Retrieved context:
        {context}

        Answer format:
        1. Summary
        2. Key supporting evidence
        3. Possible implications
        4. Sources
        5. Disclaimer: This is not financial advice.
        """

    def generate_answer(self, question: str, documents: list[dict]) -> str:
        prompt = self.build_prompt(question, documents)
        return self.llm_provider.generate(prompt)