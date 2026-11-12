"""Runs as the unprivileged sandbox user: install a seccomp filter that blocks network sockets, then exec the solution.

    python -I launcher.py solution.py

The filter survives exec and can't be removed by the solution. socket() is allowed only for AF_UNIX; io_uring (which
can open sockets without socket()) and ptrace are blocked outright.
"""

import ctypes
import os
import sys

ALLOW, EPERM = 0x7FFF0000, 0x00050000 | 1
SCMP_CMP_NE, AF_UNIX = 1, 1


class ArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


lib = ctypes.CDLL("libseccomp.so.2")
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
                                       ctypes.POINTER(ArgCmp)]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(ALLOW)
assert ctx
not_unix = (ArgCmp * 1)(ArgCmp(0, SCMP_CMP_NE, AF_UNIX, 0))
assert lib.seccomp_rule_add_array(ctx, EPERM, lib.seccomp_syscall_resolve_name(b"socket"), 1, not_unix) == 0
for name in (b"io_uring_setup", b"ptrace", b"process_vm_readv", b"process_vm_writev"):
    nr = lib.seccomp_syscall_resolve_name(name)
    if nr >= 0:
        assert lib.seccomp_rule_add_array(ctx, EPERM, nr, 0, None) == 0
assert lib.seccomp_load(ctx) == 0
os.execv(sys.executable, [sys.executable, "-I", *sys.argv[1:]])
