"""Fresh Linux process: prohibit socket syscalls before importing the reader.

The SDK provides protobuf classes only. Audit Python network/process attempts
as well, so a swallowed networking error cannot masquerade as no attempt.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    assert sys.platform == "linux"
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    library = ctypes.CDLL("libseccomp.so.2")
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # Linux SECCOMP_RET_ALLOW.
    assert context
    blocked = ["socket", "socketpair", "connect", "sendto", "sendmsg"]
    for name in blocked:
        syscall = library.seccomp_syscall_resolve_name(name.encode())
        assert syscall >= 0
        action = 0x00050000 | errno.ENETUNREACH  # Linux SECCOMP_RET_ERRNO.
        assert library.seccomp_rule_add(context, action, syscall, 0) == 0
    assert library.seccomp_load(context) == 0
    library.seccomp_release(context)
    attempts = []
    watched = {
        "socket.__new__",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.sendto",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
    }

    def audit(event: str, _arguments: object) -> None:
        if event in watched:
            attempts.append(event)
            raise RuntimeError("network/process activity during a local log read")

    sys.addaudithook(audit)
    from trainspotter import load_run

    cases = []
    for case in ["stable", "unstable-lr", "logged-null", "custom-counter"]:
        path = root / case / "run.wandb"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        run = load_run(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        cases.append(
            {
                "case": case,
                "input_sha256": before,
                "history_records": run.meta["wandb_native"]["history_records"],
                "input_complete": run.input_complete(),
            }
        )
    assert attempts == []
    receipt = {
        "method": (
            "Fresh Python process; seccomp prohibits socket/socketpair/connect/sendto/sendmsg "
            "before application imports; Python audit hooks reject network/process activity, "
            "including swallowed attempts."
        ),
        "blocked_kernel_syscalls": blocked,
        "python_network_or_process_attempts": attempts,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "Read all four native captures with kernel networking prohibited; "
        "no Python network/process attempts"
    )


if __name__ == "__main__":
    main()
