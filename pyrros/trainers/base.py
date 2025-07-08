"""
Définition d’une interface minimale que tous les trainers Pyrros doivent
implémenter.  L’objectif est double :

1.  Pouvoir typer / factoriser le code des recipes (`train_grpo_qwen.py`, etc.).
2.  Rendre chaque trainer interchangeable (maison, Composer, FSDP, …).

Un *trainer* est ici l’objet « haut niveau » qui orchestre l’entraînement ;
il peut déléguer la boucle interne à Composer, Lightning, accelerate ou
un simple script PyTorch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTrainer(ABC):
    """Contrat minimal commun à tous les trainers Pyrros."""

    @abstractmethod
    def train(self) -> None: ...

    # Les deux méthodes ci-dessous ne sont *pas* indispensables à chaque
    # implémentation ; on donne des implémentations par défaut (NOP) pour
    # garder la souplesse.
    def save_checkpoint(self, path: str | Path) -> None:  # noqa: D401
        """Enregistre un état d’entraînement (optionnel)."""
        pass

    def eval(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        """Phase d’évaluation/validation (optionnelle)."""
        pass
