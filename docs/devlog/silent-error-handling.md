# Silent error handling in the heatsim modules

**Date:** 2026-09-01
**Status:** Known gap, deliberately not fixed in the lint cleanup. Referenced from
`pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]` comment.

## The exemption

`ruff` flags 103 findings across the heatsim modules: 62 `BLE001` (blind `except Exception`)
and 41 `S110` (`try`/`except`/`pass`). They are exempted in `pyproject.toml` rather than
rewritten, and that exemption is defensible:

These handlers wrap `bpy.ops.*` calls. Blender operators raise undocumented exception types
and fail in `--background` for reasons that are not part of its public API -- a `poll()`
failure on one object's UV unwrap, a `mode_set` with no valid context, a `select_all` while
an earlier object left the context in EDIT mode. The pipeline solves whole scenes (289
objects for `visionsim50/diningroom`), so one object's operator failure must degrade **that
object** to a documented fallback, not abort the scene. Narrowing to a specific exception
type would mean guessing at Blender internals; letting them propagate would trade a
per-object degradation for total loss.

## The gap the exemption hides

**Only 4 of the 40 `S110` handlers carry any explanatory comment, and almost none log.**
That is a real defect, and this project has already paid for it twice:

| incident | what was silently swallowed | cost |
|---|---|---|
| irradiance dropped for 79% of objects | `_combine` discarded a flux array whose length did not match the node count | Days. Every headline metric looked healthy -- non-finite counts 0, `r/(sigma*T^4)` a textbook 1.0, plausible spreads. Found only when someone looked at the images and asked why the chairs were dark. |
| 231 objects demoted out of the atlas | `prepare_object_bake_uv` returned early on meshes with no authored UVs, so the atlas UV was never written | Presented as "objects look flat", diagnosed only by counting demotion warnings in the log -- which `grep -c` under-counted because the log wraps. |

Both failures produced complete, plausible-looking renders and exit code 0. That is the most
expensive failure mode available, and silent `except: pass` is its enabling condition.

## The fix, when it is done deliberately

**Log, do not remove.** The handlers are correct; their silence is not. Each should emit at
`debug` level with the object name and the operation attempted:

```python
except Exception as exc:  # bpy operators raise undocumented types; see pyproject.toml
    _log.debug("[heatsim] %s: %s failed (%s); using fallback", obj.name, "bake-UV prep", exc)
```

`debug` rather than `warning` because these fire per-object in tight loops over hundreds of
objects -- at `warning` a normal scene would emit hundreds of lines and the signal would be
lost again, for a different reason.

This was kept out of the lint cleanup on purpose. It touches 40 call sites and changes
runtime behaviour (log volume), which does not belong in a change whose stated scope is
"make the linter pass". It wants its own commit, its own review, and a check that log volume
stays sane on a 289-object scene.

## Related

- `thermal-atlas/dataset-audit-three-scenes.md` section 8 -- the flux-drop incident.
- `thermal-atlas/dataset-audit-three-scenes.md` section 10 -- the atlas-demotion incident,
  including the `grep -c` log-wrapping trap.
