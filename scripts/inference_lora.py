import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv
from tavily import TavilyClient

def get_web_context(query, tavily_client):
    print(f"🔍 Searching the web securely for: '{query}'...")
    try:
        # Search using the official Tavily API
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        context = ""
        for i, result in enumerate(response.get('results', [])):
            context += f"[Source {i+1}]: {result['content']}\n"
        
        if context:
            print("✅ Web context successfully retrieved.")
        return context
    except Exception as e:
        print(f"⚠️ Secure search failed: {e}")
        return ""

def main():
    print("="*60)
    print("🤖 CLUTCH-AI LOCAL LORA TESTER")
    print("="*60)
    
    hf_token = input("\n🔑 Please paste your HuggingFace Token (starts with hf_...): ")
    login(hf_token)
    
    # Securely load environment variables
    load_dotenv()
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("❌ Error: TAVILY_API_KEY not found in .env file.")
        return
        
    tavily_client = TavilyClient(api_key=tavily_key)
    
    # Verify the files are in the root directory
    if not os.path.exists("adapter_model.safetensors") or not os.path.exists("adapter_config.json"):
        print("❌ Error: Could not find adapter_model.safetensors or adapter_config.json in the current directory.")
        print("Make sure you put them right inside the Clutch-AI folder!")
        return

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"📥 Loading base model ({model_name}) in 4-bit mode...")
    
    # 4-bit config to fit it in normal VRAM on your computer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("🧩 Attaching your Checkpoint 500 brain (LoRA adapter)...")
    model = PeftModel.from_pretrained(base_model, ".")
    
    print("✅ Model fully loaded and ready to chat! (Secure Web Search is ON)\n")

    base_system_prompt = "You are Clutch-AI, a powerful, brilliant, and deeply analytical artificial intelligence, engineered by Battu Ujwal Reddy (also known as the Clutch Group)."

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
            
        # 1. Fetch industry-grade web context
        web_context = get_web_context(user_input, tavily_client)
        
        # 2. Safely inject context into the prompt
        if web_context:
            system_prompt = f"{base_system_prompt}\n\nAnswer the user's question using the following live web search results:\n{web_context}"
        else:
            system_prompt = base_system_prompt
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        print("\nClutch-AI:", end=" ", flush=True)
        
        with torch.no_grad():
            # Handle Llama 3 specific end-of-turn tokens
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=terminators,
                do_sample=True,
            )
            
        # Decode only the newly generated tokens
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        print(response.strip() + "\n")

if __name__ == "__main__":
    main()
