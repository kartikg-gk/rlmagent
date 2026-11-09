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

async def find_child(name):
    """Look up one of your children by name."""
    return _child_handle((await _child_operation("find", None, name))["id"])

async def child_info(handle):
    """Inspect a child's status, identity and recorded usage without consuming its output."""
    return await _child_operation("info", handle.id)

async def steer_child(handle, text, message_id=None):
    """Deliver an instruction at an active child's next model request."""
    return await _child_operation("steer", handle.id, text=text, message_id=message_id)
'''


def compatibility_helpers(call: str) -> str:
    return f'''
_child_handle = _SubAgent

async def _child_operation(action, key=None, name=None, **arguments):
    return await {call}("child_handle", {{"action": action, "id": key, "name": name, **arguments}})
''' + HELPERS
