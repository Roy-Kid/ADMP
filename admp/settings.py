#!/usr/bin/env python
from jax import jit
from jax.config import config

PRECISION = 'single' #  'double'

DO_JIT = True

if PRECISION == 'double':
    config.update("jax_enable_x64", True)

def jit_condition(*args, **kwargs):
    def jit_deco(func):
        if DO_JIT:
            return jit(func, *args, **kwargs)
        else:
            return func
    return jit_deco

# def conditional_decorator(dec, condition):
#     def decorator(func):
#         if not condition:
#             # Return the function unchanged, not decorated.
#             return func
#         return dec(func)
#     return decorator
