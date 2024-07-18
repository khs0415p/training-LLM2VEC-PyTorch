from typing import Tuple

class Prompts:
    def __init__(self, path: str) -> None:

        if "gemma" in path:
            source_prompt = """<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{user_content}"""
            target_prompt = """\n<|im_start|>assistant\n{assistant_content}<|im_end|>"""
            end_flag = '<|im_end|>'
            if "it" in path:
                source_prompt = """<start_of_turn>user\n{system_content}\n{user_content}"""
                target_prompt = """\n<start_of_turn>model\n{assistant_content}<end_of_turn>"""
                end_flag = '<end_of_turn>'
        elif "llama" in path:
            if 'meta-llama-3' in path:
                if 'instruct' in path:
                    source_prompt = """<|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}"""
                    target_prompt = """<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_content}<|eot_id|>"""
                    end_flag = '<|eot_id|>'
                else:
                    # bos 없어서 직접 넣어줌.
                    source_prompt = """<|begin_of_text|><|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{user_content}"""
                    target_prompt = """\n<|im_start|>assistant\n{assistant_content}<|im_end|>"""
                    end_flag = '<|im_end|>'
            elif 'meta-llama-2':
                source_prompt = """[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} """
                target_prompt = """ {assistant_content} """
                end_flag = '[/INST]'
        elif "mistral" in path:
            source_prompt = """[INST] {system_content}\n{user_content} """
            target_prompt = """{assistant_content}"""
            end_flag = '[/INST]'
        else:
            raise "Choose a model from gemma & llama2 & llama3 & mistral"
        
        self.source_prompt = source_prompt
        self.target_prompt = target_prompt
        self.end_flag = end_flag
        
    def get_prompts(self) -> Tuple[str, str, str]:
        return self.source_prompt, self.target_prompt, self.end_flag
    
if __name__ == "__main__":
    p = Prompts('meta-llama/meta-llama-2-8b')
    s, t, e = p.get_prompts()
    print(s, t, e)
    
    