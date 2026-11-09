"""Names and guest helpers for parent-owned child handles."""

import re


def checked_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("child name must start with a letter and contain up to 64 letters, digits, underscores or hyphens")
    return name


HELPERS = '''
async def name_child(handle, name):
    """Assign or change a name under the current parent and return the same handle."""
    await _child_operation("name", handle.id, name)
    return handle

async def child(name):
    """Look up one of your children by name."""
    return _child_handle((await _child_operation("find", None, name))["id"])

async def child_info(handle):
    """Inspect a child's status, identity and recorded usage without consuming its output."""
    return await _child_operation("info", handle.id)
'''


def compatibility_helpers(call: str) -> str:
    return f'''
_child_handle = _SubAgent

async def _child_operation(action, key=None, name=None):
    return await {call}("child_handle", {{"action": action, "id": key, "name": name}})
''' + HELPERS
