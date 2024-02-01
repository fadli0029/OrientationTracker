import jax
import jax.numpy as jnp

USE_JIT = True

def use_jit(q_func):
    """
    A decorator to turn JIT off for
    jax_quaternion_test.py
    """
    if USE_JIT:
        return jax.jit(q_func)
    else:
        return q_func

# numerical stability constant
EPS = 1e-6

@use_jit
def qmult_jax(q1, q2):
    """
    Batched version of quaternion multiplication.
    """
    if q1.ndim == 1:
        q1 = q1[None, :]
    if q2.ndim == 1:
        q2 = q2[None, :]

    q1s = q1[..., 0]
    q1v = q1[..., 1:]

    q2s = q2[..., 0]
    q2v = q2[..., 1:]

    q3s = q1s * q2s - jnp.sum(q1v * q2v, axis=-1)
    q3v = q1s[..., None] * q2v + q2s[..., None] * q1v + jnp.cross(q1v, q2v)
    res = jnp.concatenate([q3s[..., None], q3v], axis=-1)
    if res.shape[0] == 1:
        res = res[0]
    return res

@use_jit
def qinverse_jax(q):
    """
    Batched version of quaternion inverse.
    """
    q_conj = jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)
    q_norm_sq = (jnp.linalg.norm(q, axis=-1, keepdims=True)**2) + EPS
    return q_conj / q_norm_sq

@use_jit
def qexp_jax(q):
    """
    Batched version of quaternion exponential.
    """
    qv = q[..., 1:]
    qv_norm = jnp.linalg.norm(qv, axis=-1, keepdims=True) + EPS
    qv_normed = qv / qv_norm
    qv_normed *= jnp.sin(qv_norm)
    q_term = jnp.concatenate([jnp.cos(qv_norm), qv_normed], axis=-1)
    return jnp.exp(q[..., :1]) * q_term

@use_jit
def qlog_jax(q):
    """
    Batched version of quaternion logarithm.
    """
    qv = q[..., 1:]
    qv_norm = jnp.linalg.norm(qv, axis=-1, keepdims=True) + EPS
    qv_normed = qv / qv_norm

    q_norm = jnp.linalg.norm(q, axis=-1, keepdims=True) + EPS
    q_term = jnp.concatenate([jnp.log(q_norm), qv_normed * jnp.arccos(q[..., :1] / q_norm)], axis=-1)
    return q_term
