from huggingface_hub import login
import os

print("=== Configuración de HuggingFace ===\n")

# Opción A: Login interactivo
print("Por favor, ingresa tu token de acceso de HuggingFace")
print("(Lo puedes encontrar en: https://huggingface.co/settings/tokens)")
login(token="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")

print("\n✓ Autenticación exitosa!")
print("Ahora puedes descargar datasets y subir modelos.")

# Verificar que estás autenticado
from huggingface_hub import whoami

try:
    user_info = whoami()
    print(f"\nUsuario autenticado: {user_info['name']}")
    print(f"Email: {user_info.get('email', 'No disponible')}")
except Exception as e:
    print(f"Error al verificar usuario: {e}")
