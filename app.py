import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

MODEL_PATH = r"C:\Users\pc798\OneDrive\Escritorio\IA learn\list_project\modelo_IA\model_enfermeria_final"


print("cargando tokenizer...") 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("cargando modelo de enfermería...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, device_map="auto")

model.eval()
print("Model cargado exitosament. ")



def chat_enfermeria(pregunta, max_tokens=200): 
    prompt = f"""ERES una asistente experta en enfermeria universitaia. respondes de forma clara, tecnica y padagogia.
    Usuario: {pregunta} Enfermero: """

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad(): 
        output = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            temperature=0.6,
            top_p = 0.9,
            do_sample=True,
            pad_token_id= tokenizer.eos_token_id
        )
    texto = tokenizer.decode(output[0], skip_special_tokens=True)
    respuesta = texto.split("Enfermero:")[-1].strip()
    return respuesta


print("¡Listo para responder preguntas de enfermería! Escribe 'salir' para terminar.")

while True:
    pregunta = input("Tu:")
    if pregunta.lower() ==  ["salir", "exit", "quit"]:
        print("¡Hasta luego!")
        break

    respuesta = chat_enfermeria(pregunta)
    print(f"Enfermero: {respuesta}")
