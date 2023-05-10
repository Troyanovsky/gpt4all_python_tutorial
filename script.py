from llama_cpp import Llama
llm = Llama(model_path="/Users/guodongzhao/Library/Application Support/nomic.ai/GPT4All/ggml-wizardLM-7B.q4_2.bin")
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)