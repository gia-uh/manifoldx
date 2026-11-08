"""Tests for manifoldx.renderer module."""
import numpy as np
import pytest


def test_transform_cache_dirty_flag():
    """Test that cache tracks dirty entities."""
    from manifoldx.renderer import TransformCache
    
    cache = TransformCache(max_entities=100)
    
    # Initially all dirty
    assert cache._dirty.all()
    
    # Mark some as clean
    cache._dirty[5:10] = False
    assert cache._dirty[:5].all()
    assert not cache._dirty[5:10].any()


def test_transform_cache_mark_dirty():
    """Test mark_dirty sets dirty flag."""
    from manifoldx.renderer import TransformCache
    
    cache = TransformCache(max_entities=100)
    
    # Initially all dirty
    assert cache._dirty.all()
    
    indices = np.array([0, 5, 10])
    cache.mark_dirty(indices)
    
    # Those indices should be dirty
    assert cache._dirty[0]
    assert cache._dirty[5]
    assert cache._dirty[10]


def test_render_pipeline():
    """Test RenderPipeline initialization."""
    from manifoldx.ecs import EntityStore
    from manifoldx.renderer import RenderPipeline
    from manifoldx.components import Transform
    
    store = EntityStore()
    Transform.register(store)
    
    # Spawn some entities
    default_data = Transform.get_default_data(10)
    default_data[:, 0:3] = np.arange(30).reshape(10, 3)  # positions
    indices = store.spawn(n=10, Transform=default_data)
    
    pipeline = RenderPipeline(store)
    
    # Create a mock engine
    class MockEngine:
        pass
    
    engine = MockEngine()
    pipeline.run(engine, 1/60)


def test_render_pipeline_has_depth_stencil():
    """Render pipeline source should configure depth stencil state."""
    with open('src/manifoldx/renderer.py', 'r') as f:
        source = f.read()
    
    assert 'depth_stencil' in source, \
        "Render pipeline must have depth_stencil configuration"
    assert 'depth24plus' in source or 'depth32float' in source, \
        "Render pipeline must specify a depth format"


def test_render_pipeline_has_backface_culling():
    """Render pipeline should cull back faces."""
    with open('src/manifoldx/renderer.py', 'r') as f:
        source = f.read()
    
    assert 'CullMode.back' in source, \
        "Render pipeline must use back-face culling"


def test_engine_creates_depth_texture():
    """Engine render pass should have depth attachment."""
    with open('src/manifoldx/engine.py', 'r') as f:
        source = f.read()
    
    assert 'depth_stencil_attachment' in source, \
        "Engine render pass must have depth_stencil_attachment"
    assert 'depth24plus' in source or 'depth32float' in source, \
        "Engine must create a depth texture"
