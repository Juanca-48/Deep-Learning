# Crear archivo: test_installation.py
import sys

print("=== Verificando instalación de bibliotecas ===\n")

required_packages = {
    'transformers': 'transformers',
    'datasets': 'datasets',
    'trl': 'trl',
    'torch': 'torch',
    'gradio': 'gradio',
    'huggingface_hub': 'huggingface_hub'
}

for package_name, import_name in required_packages.items():
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Desconocida')
        print(f"✓ {package_name:20s} v{version}")
    except ImportError:
        print(f"✗ {package_name:20s} NO INSTALADO")

# Verificar PyTorch CUDA (GPU) disponibilidad
import torch
print(f"\nPyTorch CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("Entrenamiento será en CPU (más lento pero funcional)")

print("\n=== Instalación completa ===")
