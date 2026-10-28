# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Special-mechanism component package (the ``convert`` execution layer).

Components self-declare their plugin name and the model families they belong to
via the :func:`component` decorator.  ``configs`` stays pure data; the engine
resolves a model's components at runtime through
:func:`component_names_for_model` / :func:`instantiate_component_for_model_and_name`.  Concrete
component classes + their hardware executors live in per-family modules (e.g.
``deepseek_v4.py`` / ``deepseek_v4_1.py``) and self-register here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs import ModelType
    from ..recipes import RecipeContext


class QuantComponent:
    """Special-mechanism component interface."""

    name: str = ""

    def should_trigger(self, filters: dict, ctx: RecipeContext) -> bool:
        raise NotImplementedError

    def build_converters(self, filters: dict, ctx: RecipeContext) -> list:
        raise NotImplementedError


_COMPONENT_TYPES: dict[tuple[ModelType, str], type[QuantComponent]] = {}
# Per-model component names, kept in registration order (list, not a set, so
# iteration order is deterministic across processes -- a set would randomize via
# PYTHONHASHSEED and break reproducible converter ordering).
_MODEL_COMPONENT_NAMES: dict[ModelType, list[str]] = {}


def component(name: str, *, models: tuple[ModelType, ...]):
    """Register a :class:`QuantComponent` subclass under its plugin name and the
    model families it is available to.

    The plugin ``name`` denotes the feature; several classes may share the same
    ``name`` as long as their ``models`` sets are disjoint.  The decorator is the
    single place that declares the model -> component association; ``configs`` no
    longer lists components per model.
    """
    if not name:
        raise ValueError("component name must be a non-empty string")
    if not models:
        raise ValueError(f"component {name!r} must declare at least one model")

    def deco(cls: type[QuantComponent]) -> type[QuantComponent]:
        for kind in set(models):
            key = (kind, name)
            if key in _COMPONENT_TYPES:
                raise ValueError(
                    f"component {name!r} already registered for model {kind.value!r} "
                    f"as {_COMPONENT_TYPES[key].__qualname__}"
                )
            _COMPONENT_TYPES[key] = cls
            # Append-only, dedupced so a model's list never repeats a name and
            # keeps pure registration order.
            names = _MODEL_COMPONENT_NAMES.setdefault(kind, [])
            if name not in names:
                names.append(name)
        cls.name = name
        return cls

    return deco


def instantiate_component_for_model_and_name(kind: ModelType, name: str) -> QuantComponent:
    """Reflect-instantiate a model's component plugin by its feature name."""
    try:
        return _COMPONENT_TYPES[(kind, name)]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported component plugin {name!r} for model {kind.value!r}; "
            f"available: {sorted(feature for (model, feature) in _COMPONENT_TYPES if model is kind)}"
        ) from exc


def component_names_for_model(kind: ModelType) -> tuple[str, ...]:
    """Component names a model family owns, in registration order."""
    return tuple(_MODEL_COMPONENT_NAMES.get(kind, ()))


# Import the family component modules here (after ``QuantComponent``/``component``
# are defined) so their ``@component`` decorators self-register into the registry
# above.  Nothing imports ``components.deepseek_v4_1`` on its own, so without this
# explicit activation the V4.1 components would never be registered.
def _register_components():
    from . import deepseek_v4 as _deepseek_v4  # noqa: F401  (self-registering)
    from . import deepseek_v4_1 as _deepseek_v4_1  # noqa: F401  (self-registering)


_register_components()
