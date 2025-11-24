"""
APCI (Asistentul Personalizat de Cercetare și Învățare)
Sistemul principal pentru procesarea documentelor și generarea de răspunsuri inteligente
"""

from .rag_module_flash import APCISystem, RAGConfig, create_apci_system

__version__ = "1.0.0"
__author__ = "APCI Development Team"

__all__ = [
    'APCISystem',
    'RAGConfig', 
    'create_apci_system'
]
