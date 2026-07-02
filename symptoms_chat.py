# symptoms_chat.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Toggle for testing vs real model
TEST_MODE = True  

# Global chatbot variable
chatbot = None

def init_chatbot():
    """Initialize chatbot depending on TEST_MODE."""
    global chatbot
    if TEST_MODE:
        print("🧪 TEST_MODE enabled: Using dummy responses instead of downloading Llama model.")
        chatbot = None
        return

    # Hugging Face model setup
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    print("⏳ Loading real Llama-3 model... This may take hours the first time.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    chatbot = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("✅ Llama-3 loaded successfully.")


def emergency_check(user_input: str) -> bool:
    """Check if input contains emergencies."""
    emergencies = [
        "chest pain", "difficulty breathing", 
        "shortness of breath", "severe bleeding"
    ]
    return any(term in user_input.lower() for term in emergencies)


def run_symptom_checker(user_input: str) -> str:
    """Main entry for Flask app to use chatbot."""
    if emergency_check(user_input):
        return "⚠️ Emergency detected: Please call your local emergency services immediately."

    if TEST_MODE:
        # Fake response for testing
        return f"(TEST_MODE) You said: {user_input}. This is a dummy medical response."

    # Build real prompt
    system_prompt = (
        "You are a helpful and safe medical assistant. "
        "Provide accurate but non-diagnostic information. "
        "Encourage users to seek professional help for serious conditions."
    )
    prompt = f"<|begin_of_text|><|system|>{system_prompt}<|end|><|user|>{user_input}<|end|><|assistant|>"

    response = chatbot(
        prompt,
        max_new_tokens=150,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )[0]['generated_text']

    # Clean output
    return response.split("<|assistant|>")[-1].strip()


# Initialize at import
init_chatbot()
