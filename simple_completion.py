# Original repo for the python library: https://github.com/abetlen/llama-cpp-python
# Install the library first with: pip install llama-cpp-python

# Import the Llama library.
from llama_cpp import Llama

# Create a Llama object. Modify the model_path to point to your model.
# llama.cpp supports mostly GGML models running on CPU
llm = Llama(model_path="/Users/guodongzhao/Library/Application Support/nomic.ai/GPT4All/ggml-wizardLM-7B.q4_2.bin")

# Add the input prompt. The stop parameter should match what you have in the input prompt.
# Different models may have different stop words because of their training dataset, refer to the model card for more information.
output = llm("Q: Explain AI in one sentence. A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)

# Get the text from the output.
output_string = output['choices'][0]['text']

# Print the output string.
print(output_string)