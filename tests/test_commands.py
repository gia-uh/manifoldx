"""Tests for manifoldx.commands module."""
import numpy as np
import pytest


def test_command_buffer_append():
    """CommandBuffer should allow appending commands."""
    from manifoldx.commands import CommandBuffer, Command, CommandType
    buf = CommandBuffer()
    cmd = Command(CommandType.UPDATE_COMPONENT, 
                  {'component_name': 'pos', 'indices': [0,1], 'new_data': np.ones((2,3))})
    buf.append(cmd)
    assert len(buf) == 1


def test_command_buffer_clear():
    """CommandBuffer should clear on command."""
    from manifoldx.commands import CommandBuffer, Command, CommandType
    buf = CommandBuffer()
    buf.append(Command(CommandType.NOP, {}))
    buf.append(Command(CommandType.NOP, {}))
    buf.clear()
    assert len(buf) == 0


def test_command_types():
    """Verify command types are correctly numbered."""
    from manifoldx.commands import CommandType
    assert CommandType.NOP == 0
    assert CommandType.SPAWN == 1
    assert CommandType.DESTROY == 2
    assert CommandType.UPDATE_COMPONENT == 3


def test_command_execution_order():
    """Test that commands execute in order: spawn → update → destroy."""
    from manifoldx.commands import CommandBuffer, Command, CommandType
    from manifoldx.ecs import EntityStore
    
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    buf = CommandBuffer()
    
    # Spawn entities
    buf.append(Command(CommandType.SPAWN, 
                      {'n': 3, 'components': {'position': np.ones((3,3))}}))
    
    # Update position
    indices = np.array([0, 1, 2])
    buf.append(Command(CommandType.UPDATE_COMPONENT,
                      {'component_name': 'position', 'indices': indices, 
                       'new_data': np.full((3,3), 5.0)}))
    
    # Destroy
    buf.append(Command(CommandType.DESTROY, {'indices': indices}))
    
    # Execute
    buf.execute(store)
    
    # Final state: entities should be dead
    assert not store._alive[indices].any()
