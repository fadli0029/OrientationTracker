# Test batch implementation of quaternion operations
# by comparing to transforms3d.quaternions

import jax
import time
from jax import random
from jax_quaternion import *
import transforms3d.quaternions as tq

key = random.PRNGKey(0)
ATOL = 1e-6
RTOL = 1e-6

def random_q_batch(n=1000):
    q = jax.random.uniform(key, shape=(n, 4), dtype=jnp.float32)
    q /= jnp.linalg.norm(q, axis=1).reshape(-1, 1)
    return q

def random_q():
    q = jax.random.uniform(key, shape=(4,), dtype=jnp.float32)
    q /= jnp.linalg.norm(q)
    return q

def test_qmult():
    # single quaternions
    q1 = random_q()
    q2 = random_q()
    q12 = qmult_jax(q1, q2)
    tq12 = tq.qmult(q1, q2)
    if not jnp.allclose(q12, tq12, atol=ATOL, rtol=RTOL):
        print(f'Failed in single quaternion case: {q1}, {q2}')
        print(f'qmult_jax: {q12}')
        print(f'transforms3d: {tq12}')

    # batch of quaternions
    q1 = random_q_batch()
    q2 = random_q_batch()
    start = time.time()
    q12 = qmult_jax(q1, q2)
    print(f'qmult_jax time: {(time.time() - start):.3f}')
    start = time.time()
    for i in range(q1.shape[0]):
        tq12 = tq.qmult(q1[i], q2[i])
        if not jnp.allclose(q12[i], tq12, atol=ATOL, rtol=RTOL):
            print(f'Failed in batch case: {q1[i]}, {q2[i]}')
            print(f'qmult_jax: {q12[i]}')
            print(f'transforms3d: {tq12}')
    print(f'transforms3d time: {(time.time() - start):.3f}')
    print('PASSED: qmult')

def test_qinverse():
    # single quaternions
    q = random_q()
    qi = qinverse_jax(q)
    tqi = tq.qinverse(q)
    if not jnp.allclose(qi, tqi, atol=ATOL, rtol=RTOL):
        print(f'Failed in single quaternion case: {q}')
        print(f'qinverse_jax: {qi}')
        print(f'transforms3d: {tqi}')

    # batch of quaternions
    q = random_q_batch()
    start = time.time()
    qi = qinverse_jax(q)
    print(f'qinverse_jax time: {(time.time() - start):.3f}')
    start = time.time()
    for i in range(q.shape[0]):
        tqi = tq.qinverse(q[i])
        if not jnp.allclose(qi[i], tqi, atol=ATOL, rtol=RTOL):
            print(f'Failed in batch case: {q[i]}')
            print(f'qinverse_jax: {qi[i]}')
            print(f'transforms3d: {tqi}')
    print(f'transforms3d time: {(time.time() - start):.3f}')
    print('PASSED: qinverse')

def test_qexp():
    # single quaternions
    q = random_q()
    qe = qexp_jax(q)
    tqe = tq.qexp(q)
    if not jnp.allclose(qe, tqe, atol=ATOL, rtol=RTOL):
        print(f'Failed in single quaternion case: {q}')
        print(f'qexp_jax: {qe}')
        print(f'transforms3d: {tqe}')

    # batch of quaternions
    q = random_q_batch()
    start = time.time()
    qe = qexp_jax(q)
    print(f'qexp_jax time: {(time.time() - start):.3f}')
    start = time.time()
    for i in range(q.shape[0]):
        tqe = tq.qexp(q[i])
        if not jnp.allclose(qe[i], tqe, atol=ATOL, rtol=RTOL):
            print(f'Failed in batch case: {q[i]}')
            print(f'qexp_jax: {qe[i]}')
            print(f'transforms3d: {tqe}')
    print(f'transforms3d time: {(time.time() - start):.3f}')
    print('PASSED: qexp')

def test_qlog():
    # single quaternions
    q = random_q()
    ql = qlog_jax(q)
    tql = tq.qlog(q)
    if not jnp.allclose(ql, tql, atol=ATOL, rtol=RTOL):
        print(f'Failed in single quaternion case: {q}')
        print(f'qlog_jax: {ql}')
        print(f'transforms3d: {tql}')

    # batch of quaternions
    q = random_q_batch()
    start = time.time()
    ql = qlog_jax(q)
    print(f'qlog_jax time: {(time.time() - start):.3f}')
    start = time.time()
    for i in range(q.shape[0]):
        tql = tq.qlog(q[i])
        if not jnp.allclose(ql[i], tql, atol=ATOL, rtol=RTOL):
            print(f'Failed in batch case: {q[i]}')
            print(f'qlog_jax: {ql[i]}')
            print(f'transforms3d: {tql}')
    print(f'transforms3d time: {(time.time() - start):.3f}')
    print('PASSED: qlog')

# Run all tests
def run_tests():
    test_qmult()
    print("")
    test_qinverse()
    print("")
    test_qexp()
    print("")
    test_qlog()
