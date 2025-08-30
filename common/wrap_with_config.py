from __future__ import annotations
import inspect
from dataclasses import is_dataclass, asdict
from typing import Any, Mapping, Callable


def _to_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort convert obj to a plain dict of fields->values."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return dict(obj.model_dump())
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return dict(obj.dict())
        except Exception:
            pass
    # Generic Python object with attributes
    return {k: getattr(obj, k) for k in dir(obj)
            if not k.startswith("_") and hasattr(obj, k) and not callable(getattr(obj, k))}


def call_with_config(
    func: Callable[..., Any],
    config: Any,
    *,
    aliases: Mapping[str, str] | None = None,
    allow_extras_to_kwargs: bool = True,
) -> Any:
    """
    Call `func` by pulling arguments from `config`.

    - `config` can be a dataclass, dict, Pydantic model, or any object with public attrs.
    - `aliases`: mapping from source field name in config -> destination param name in func.
    - `allow_extras_to_kwargs`: if func has **kwargs, pass leftover fields to it.

    Raises:
        TypeError if required parameters are missing from config.
    """
    data = _to_mapping(config)

    # Apply aliases (source -> destination), without clobbering explicit destinations
    if aliases:
        for src, dst in aliases.items():
            if src in data and dst not in data:
                data[dst] = data[src]

    sig = inspect.signature(func)
    params = sig.parameters

    # Partition parameters by kind
    positional_only = [name for name, p in params.items() if p.kind == inspect.Parameter.POSITIONAL_ONLY]
    pos_or_kw = [name for name, p in params.items() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    kw_only = [name for name, p in params.items() if p.kind == inspect.Parameter.KEYWORD_ONLY]
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # Build args (for positional-only) and kwargs
    args = []
    kwargs = {}

    # Fill positional-only by name (must exist in config; cannot be passed as keywords later)
    missing_pos_only = []
    for name in positional_only:
        p = params[name]
        if name in data:
            args.append(data[name])
        else:
            if p.default is inspect._empty:
                missing_pos_only.append(name)
            # If default exists, we just skip; function will use default
    if missing_pos_only:
        raise TypeError(f"{func.__name__} missing required positional-only params from config: {missing_pos_only}")

    # Fill positional-or-keyword and keyword-only from config (as kwargs)
    for name in pos_or_kw + kw_only:
        if name in data:
            kwargs[name] = data[name]

    # Compute missing required among those that must come from config (pos-or-kw and kw-only)
    missing_required = [
        name for name in pos_or_kw + kw_only
        if name not in kwargs
        and params[name].default is inspect._empty
    ]
    if missing_required:
        raise TypeError(f"{func.__name__} missing required params from config: {missing_required}")

    # Pass extras to **kwargs if allowed and available
    if has_varkw and allow_extras_to_kwargs:
        used = set(positional_only) | set(kwargs.keys())
        extras = {k: v for k, v in data.items() if k not in used}
        # Don't accidentally pass alias sources if they map to the same value
        kwargs.update(extras)

    return func(*args, **kwargs)

# -----------------------------
# Examples / quick self-test
# -----------------------------
if __name__ == "__main__":
    from dataclasses import dataclass

    # Example 1: dataclass -> function with matching names
    @dataclass
    class ConfigA:
        arg1: int = 10
        arg2: int = 20
        extra: str = "ignored"

    def funcA(arg1, arg2):
        print("funcA:", arg1, arg2)

    call_with_config(funcA, ConfigA())  # funcA: 10 20

    # Example 2: different signature (subset)
    def funcB(arg2):
        print("funcB:", arg2)

    call_with_config(funcB, ConfigA())  # funcB: 20

    # Example 3: keyword-only and defaults
    def funcC(x, *, y=5, z=7):
        print("funcC:", x, y, z)

    @dataclass
    class CfgC:
        x: int = 1
        z: int = 42

    call_with_config(funcC, CfgC())  # funcC: 1 5 42

    # Example 4: positional-only (Python 3.8+ syntax via "/" in def)
    # Note: define via exec for compatibility in older interpreters lacking "/"
    ns = {}
    exec(
        "def funcD(a, /, b, *, c=0):\n"
        "    print('funcD:', a, b, c)\n",
        ns
    )
    funcD = ns["funcD"]

    @dataclass
    class CfgD:
        a: int = 99
        b: int = 1
        c: int = 2

    call_with_config(funcD, CfgD())  # funcD: 99 1 2

    # Example 5: aliases (config name -> function param name)
    def train(lr, batch_size):
        print("train:", lr, batch_size)

    @dataclass
    class TrainCfg:
        learning_rate: float = 3e-4
        bs: int = 128

    call_with_config(train, TrainCfg(), aliases={"learning_rate": "lr", "bs": "batch_size"})  # train: 0.0003 128

    # Example 6: plain dict and **kwargs passthrough
    def funcE(x, **kwargs):
        print("funcE:", x, kwargs)

    cfgE = {"x": 1, "foo": "bar", "baz": 3}
    call_with_config(funcE, cfgE)  # funcE: 1 {'foo': 'bar', 'baz': 3}

    # Example 7: Pydantic model (both v1 and v2 are handled if available)
    try:
        from pydantic import BaseModel
        class PM(BaseModel):
            a: int = 7
            b: int = 8
        def funcF(a, b):
            print("funcF:", a, b)
        call_with_config(funcF, PM())  # funcF: 7 8
    except Exception:
        pass
