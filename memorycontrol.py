"""This holds a routine for restricting the current process memory on Windows."""
import multiprocessing
import ctypes


def set_memory_limit(memory_limit):
    """Creates a new unnamed job object and assigns the current process to it.
    The job object will have the given memory limit in bytes: the given process
    together with its descendant processes will not be allowed to exceed
    the limit. If purge_pid_on_exit is true, when the *calling* process exits
    (the calling process can be the same or different from the given process),
    the given process and all its descendant processes will be killed."""

    import os
    pid = os.getpid()
    purge_pid_on_exit = True

    # Windows API constants, used for OpenProcess and SetInformationJobObject.
    PROCESS_TERMINATE = 0x1
    PROCESS_SET_QUOTA = 0x100
    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        """Windows API structure, used as input to SetInformationJobObject."""

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", ctypes.c_uint32),
                        ("MinimumWorkingSetSize", ctypes.c_void_p),
                        ("MaximumWorkingSetSize", ctypes.c_void_p),
                        ("ActiveProcessLimit", ctypes.c_uint32),
                        ("Affinity", ctypes.c_void_p),
                        ("PriorityClass", ctypes.c_uint32),
                        ("SchedulingClass", ctypes.c_uint32)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                        ("WriteOperationCount", ctypes.c_uint64),
                        ("OtherOperationCount", ctypes.c_uint64),
                        ("ReadTransferCount", ctypes.c_uint64),
                        ("WriteTransferCount", ctypes.c_uint64),
                        ("OtherTransferCount", ctypes.c_uint64)]

        _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_void_p),
                    ("JobMemoryLimit", ctypes.c_void_p),
                    ("PeakProcessMemoryUsed", ctypes.c_void_p),
                    ("PeakJobMemoryUsed", ctypes.c_void_p)]


    job_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    job_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY
    if purge_pid_on_exit:
        job_info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    job_info.JobMemoryLimit = memory_limit

    kernel = ctypes.windll.kernel32
    job = kernel.CreateJobObjectA(None, None)
    if job == 0:
        raise RuntimeError("CreateJobObjectA failed")
    keep_job_handle = False
    try:
        if not kernel.SetInformationJobObject(
                   job,
                   JobObjectExtendedLimitInformation,
                   ctypes.POINTER(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)(job_info),
                   ctypes.sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)):
            raise RuntimeError("SetInformationJobObject failed")

        process = kernel.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE,False, pid)
        if process == 0:
            raise RuntimeError("OpenProcess failed")
        try:
            if not kernel.AssignProcessToJobObject(job, process):
                raise RuntimeError("AssignProcessToJobObject failed")

            # If purge_pid_on_exit is true, we kill process pid and all its
            # descendants when the job handle is closed. So, we keep the handle
            # dangling, and it will be closed when *this* process terminates.
            keep_job_handle = purge_pid_on_exit
        finally:
            if not kernel.CloseHandle(process):
                raise RuntimeError("CloseHandle failed")
    finally:
        if not (keep_job_handle or kernel.CloseHandle(job)):
            raise RuntimeError("CloseHandle failed")


def allocate(bytes):
    import numpy
    try:
        result = numpy.zeros(shape=(bytes,), dtype='i1')
        print("allocation done:", bytes)
    except Exception as ex:
        print("Failed to allocate:", ex)
        raise


def runner(thunk, memory_limit, *args):
    set_memory_limit(memory_limit)
    thunk(*args)


def run_in_process_with_memory_limit(thunk, memory_limit, test_bytes):
    p = multiprocessing.Process(target=runner, args=(thunk, memory_limit, test_bytes))
    p.start()
    p.join()


def main():
    memory_limit = 1000*1000*100
    run_in_process_with_memory_limit(allocate, memory_limit=memory_limit, test_bytes=memory_limit)


if __name__ == "__main__":
    main()
