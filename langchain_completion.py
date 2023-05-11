# Your python version should be higher than 3.8.0
# install llama-cpp-python: pip install llama-cpp-python
# install langchain: pip install langchain

from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="G:\gpt_4_all_v2\gpt_4_all_v2\\bin\ggml-wizardLM-7B.q4_2.bin"
)

output = llm("Q: Explain AI in one sentence. A: ")

print(output)