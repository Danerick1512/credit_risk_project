"""Capa de Inteligencia: agrupación de módulos inteligentes.

Exporta interfaces de los módulos de lógica difusa y machine learning e integración híbrida.
"""
from src.fuzzy_module import explicar_riesgo
from src.ml_module import prob_a_tres_clases
from .integration import hybrid_decision

__all__ = ["explicar_riesgo", "prob_a_tres_clases", "hybrid_decision"]
