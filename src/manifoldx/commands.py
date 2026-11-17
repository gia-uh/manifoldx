"""Command buffer for deferred execution."""

import numpy as np
from typing import Any


# =============================================================================
# Command Types
# =============================================================================


class CommandType:
    """Command type constants."""

    NOP = 0
    SPAWN = 1  # Create new entities with component data
    DESTROY = 2  # Mark entities as dead
    UPDATE_COMPONENT = 3  # Apply computed values to component arrays
    CALL = 4  # Call a function: a deferred engine call, applied in call order


# =============================================================================
# Command
# =============================================================================


class Command:
    """Command to be executed by the command buffer."""

    def __init__(self, type: int, data: dict):
        self.type = type
        self.data = data


# =============================================================================
# CommandBuffer
# =============================================================================


class CommandBuffer:
    """Accumulator for frame commands."""

    def __init__(self, capacity: int = 10_000):
        self._commands: list[Command] = []
        self._capacity = capacity

    def append(self, cmd: Command):
        """Add a command to the buffer."""
        self._commands.append(cmd)

    def extend(self, other: "CommandBuffer") -> None:
        """Append another buffer's commands, in their order, after these."""
        self._commands.extend(other._commands)

    def clear(self):
        """Clear all commands."""
        self._commands.clear()

    def __len__(self) -> int:
        return len(self._commands)

    def execute(self, store):
        """Execute all commands in order, modifying entity store.

        A CALL may append further commands while it runs (a deferred
        `EntityHandle.destroy` queues its DESTROY); those run in the same
        pass, after it, in call order.
        """
        i = 0
        while i < len(self._commands):
            self._execute_command(self._commands[i], store)
            i += 1

    def _execute_command(self, cmd: Command, store):
        """Execute a single command."""
        if cmd.type == CommandType.SPAWN:
            n = cmd.data["n"]
            components = cmd.data["components"]
            store.spawn(n, **components)

        elif cmd.type == CommandType.DESTROY:
            indices = cmd.data["indices"]
            store.destroy(indices)

        elif cmd.type == CommandType.UPDATE_COMPONENT:
            component_name = cmd.data["component_name"]
            indices = cmd.data["indices"]
            new_data = cmd.data["new_data"]
            col_start = cmd.data.get("col_start")
            col_end = cmd.data.get("col_end")
            if col_start is not None and col_end is not None:
                store._components[component_name][indices, col_start:col_end] = new_data
            else:
                store._components[component_name][indices] = new_data

        elif cmd.type == CommandType.CALL:
            queued = cmd.data["origin"]
            try:
                cmd.data["fn"](*cmd.data["args"], **cmd.data["kwargs"])
            except Exception as error:
                if queued:
                    error.add_note(f"in deferred call queued by system {queued}")
                raise

        elif cmd.type == CommandType.NOP:
            pass


__all__ = [
    "CommandType",
    "Command",
    "CommandBuffer",
]
