# -------------------------------------------------------------------------
# Author: Muhammad Fadli Alim Arsani
# Email: fadlialim0029[at]gmail.com
# File: jax_quaternion_test.py
# Description: This file contains the test for the quaternion operations
#              implemented in jax_quaternion.py. The test is done using
#              pytest.
# Misc: This is also part of one of the projects in the course
#       "Sensing and Estimation in Robotics" taught by Prof. Nikolay
#       Atanasov @UC San Diego.
#       https://natanaso.github.io/ece276a/index.html
# -------------------------------------------------------------------------

import pytest
import jax
from jax import random
import jax.numpy as jnp
import transforms3d.quaternions as tq
from jax_quaternion import qmult_jax, qinverse_jax, qexp_jax, qlog_jax

from jax_quaternion import USE_JIT
USE_JIT = True # Disable JIT for testing

ATOL = 1e-5
RTOL = 1e-5

def random_q_batch(key, n=1000):
    q = jax.random.uniform(key, shape=(n, 4), dtype=jnp.float32)
    q /= jnp.linalg.norm(q, axis=1).reshape(-1, 1)
    return q

def random_q(key):
    q = jax.random.uniform(key, shape=(4,), dtype=jnp.float32)
    q /= jnp.linalg.norm(q)
    return q

@pytest.fixture
def key():
    return random.PRNGKey(0)

def test_qmult(key):
    # Test single quaternion multiplication
    q1 = random_q(key)
    q2 = random_q(key)
    result_jax = qmult_jax(q1, q2)
    result_tq = tq.qmult(q1, q2)
    try:
        assert jnp.allclose(result_jax, result_tq, atol=ATOL, rtol=RTOL)
    except AssertionError:
        print("Failed for single qmult operation")
        print("qmult_jax output:", result_jax)
        print("tq.qmult output:", result_tq)
        raise

    # Test batch of quaternion multiplications
    q1_batch = random_q_batch(key)
    q2_batch = random_q_batch(key)
    result_jax_batch = qmult_jax(q1_batch, q2_batch)
    for i in range(q1_batch.shape[0]):
        result_tq_batch = tq.qmult(q1_batch[i], q2_batch[i])
        try:
            assert jnp.allclose(result_jax_batch[i], result_tq_batch, atol=ATOL, rtol=RTOL)
        except AssertionError:
            print(f"Failed at index {i} for batch qmult operation")
            print("qmult_jax output:", result_jax_batch[i])
            print("tq.qmult output:", result_tq_batch)
            raise

@pytest.mark.parametrize("function_jax, function_tq", [
    (qinverse_jax, tq.qinverse),
    (qexp_jax, tq.qexp),
    (qlog_jax, tq.qlog),
])
def test_single_quaternion_operations(function_jax, function_tq, key):
    # Test single quaternion operation
    q = random_q(key)
    result_jax = function_jax(q)
    result_tq = function_tq(q)
    try:
        assert jnp.allclose(result_jax, result_tq, atol=ATOL, rtol=RTOL)
    except AssertionError:
        print(f"Failed for single {function_jax.__name__} operation")
        print(f"{function_jax.__name__} output:", result_jax)
        print(f"{function_tq.__name__} output:", result_tq)
        raise

    # Test batch of quaternion operations
    q_batch = random_q_batch(key)
    for i in range(q_batch.shape[0]):
        result_jax_batch = function_jax(q_batch[i])
        result_tq_batch = function_tq(q_batch[i])
        try:
            assert jnp.allclose(result_jax_batch, result_tq_batch, atol=ATOL, rtol=RTOL)
        except AssertionError:
            print(f"Failed at index {i} for batch {function_jax.__name__} operation")
            print(f"{function_jax.__name__} output:", result_jax_batch)
            print(f"{function_tq.__name__} output:", result_tq_batch)
            raise
