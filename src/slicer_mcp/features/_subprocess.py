"""TotalSegmentator subprocess management helpers for Slicer code generation.

Provides reusable code-string builders for running TotalSegmentator as a
subprocess from within Slicer-bound Python code templates.  These helpers
handle the ``resource_tracker`` hang workaround (``start_new_session`` +
``killpg``) and graceful process-group termination.
"""

import json

from slicer_mcp.features.spine.constants import SPINE_SEGMENTATION_TIMEOUT

__all__ = [
    "_build_totalseg_subprocess_block",
    "_kill_process_group_code",
]


def _kill_process_group_code(
    proc_var: str = "proc",
    os_mod: str = "os",
    signal_mod: str = "signal",
    time_mod: str = "time",
    logging_alias: str = "_kpg_logging",
    indent: str = "",
) -> str:
    """Return a code snippet that gracefully kills a subprocess process group.

    Generates inline Python code for use inside Slicer-bound code templates.
    The pattern is: SIGTERM -> sleep 1s -> SIGKILL -> proc.kill() fallback.
    Catches ``ProcessLookupError`` and ``OSError`` silently; logs a warning
    for ``PermissionError``.

    Args:
        proc_var: Variable name of the subprocess.Popen instance.
        os_mod: Module name/alias for ``os`` in the generated code.
        signal_mod: Module name/alias for ``signal`` in the generated code.
        time_mod: Module name/alias for ``time`` in the generated code.
        logging_alias: Alias for ``logging`` import in the generated code.
        indent: Whitespace prefix for every generated line.

    Returns:
        Multi-line Python code string (no trailing newline).
    """
    i = indent
    pid = f"{{{proc_var}.pid}}"
    lines = [
        f"{i}if {proc_var} is not None and {proc_var}.poll() is None:",
        f"{i}    try:",
        f"{i}        {os_mod}.killpg({os_mod}.getpgid({proc_var}.pid), {signal_mod}.SIGTERM)",
        f"{i}    except ProcessLookupError:",
        f"{i}        pass",
        f"{i}    except PermissionError:",
        f"{i}        import logging as {logging_alias}",
        f'{i}        {logging_alias}.getLogger("slicer-mcp").warning(',
        f'{i}            "Cannot kill TotalSegmentator process group"',
        f'{i}            f" (PID {pid}): permission denied"',
        f"{i}        )",
        f"{i}    except OSError:",
        f"{i}        pass",
        f"{i}    {time_mod}.sleep(1)",
        f"{i}    if {proc_var}.poll() is None:",
        f"{i}        try:",
        f"{i}            {os_mod}.killpg({os_mod}.getpgid({proc_var}.pid), {signal_mod}.SIGKILL)",
        f"{i}        except ProcessLookupError:",
        f"{i}            pass",
        f"{i}        except PermissionError:",
        f"{i}            import logging as {logging_alias}",
        f'{i}            {logging_alias}.getLogger("slicer-mcp").warning(',
        f'{i}                "Cannot kill TotalSegmentator process group"',
        f'{i}                f" (PID {pid}): permission denied"',
        f"{i}            )",
        f"{i}        except OSError:",
        f"{i}            pass",
        f"{i}    try:",
        f"{i}        {proc_var}.kill()",
        f"{i}    except (ProcessLookupError, PermissionError, OSError):",
        f"{i}        pass",
    ]
    return "\n".join(lines)


def _build_totalseg_subprocess_block(
    volume_var: str = "volume_node",
    seg_var: str = "seg_node",
    task: str = "total",
    timeout_s: int = SPINE_SEGMENTATION_TIMEOUT - 60,
) -> str:
    """Build Python code block for TotalSegmentator subprocess auto-segmentation.

    Generates code that checks if ``seg_node_id`` is already set (reuse existing
    segmentation) or runs TotalSegmentator as a subprocess with the
    ``resource_tracker`` hang workaround (``start_new_session`` + ``killpg``).

    The generated code assumes ``{volume_var}`` and ``seg_node_id`` are already
    defined, and sets ``{seg_var}`` as the output segmentation node.

    Args:
        volume_var: Variable name of the input volume in the generated code.
        seg_var: Variable name for the output segmentation node.
        task: TotalSegmentator task name (e.g. ``"total"``, ``"total_mr"``).
        timeout_s: Subprocess timeout in seconds (default: SPINE_SEGMENTATION_TIMEOUT - 60).

    Returns:
        Python code string for embedding in Slicer exec code.
    """
    safe_task = json.dumps(task)

    return f"""
# --- Segmentation: reuse existing or auto-segment via subprocess ---
_seg_was_provided = seg_node_id is not None
if seg_node_id:
    {seg_var} = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not {seg_var}:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    import time as _ts_time
    import os as _ts_os
    import subprocess as _ts_subprocess
    import signal as _ts_signal
    import shutil as _ts_shutil
    import sysconfig as _ts_sysconfig

    {seg_var} = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    {seg_var}.SetName({volume_var}.GetName() + '_auto_seg')

    _ts_proc = None
    _ts_tempFolder = None

    try:
        _ts_tempFolder = slicer.util.tempDirectory()
        _ts_inputFile = _ts_os.path.join(_ts_tempFolder, "ts-input.nii")
        _ts_outputFile = _ts_os.path.join(_ts_tempFolder, "segmentation.nii")
        _ts_outputFolder = _ts_os.path.join(_ts_tempFolder, "segmentation")

        # Export volume to NIfTI
        _ts_storageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        _ts_storageNode.SetFileName(_ts_inputFile)
        _ts_storageNode.UseCompressionOff()
        _ts_writeOk = _ts_storageNode.WriteData({volume_var})
        _ts_storageNode.UnRegister(None)
        if not _ts_writeOk:
            raise RuntimeError(f"Failed to export volume to NIfTI: {{_ts_inputFile}}")

        # Build TotalSegmentator CLI command
        _ts_pythonSlicer = _ts_shutil.which('PythonSlicer')
        if not _ts_pythonSlicer:
            raise RuntimeError("PythonSlicer not found in PATH")

        _ts_exec = _ts_shutil.which('TotalSegmentator') or _ts_shutil.which('totalsegmentator')
        if not _ts_exec:
            # Fallback: check sysconfig scripts directory
            _ts_exec = _ts_os.path.join(_ts_sysconfig.get_path('scripts'), 'TotalSegmentator')
        if not _ts_os.path.isfile(_ts_exec):
            raise RuntimeError(
                "TotalSegmentator not found. Install via: "
                "slicer.util.pip_install('TotalSegmentator') or "
                "install the TotalSegmentator Slicer extension."
            )

        _ts_cmd = [_ts_pythonSlicer, _ts_exec,
                    "-i", _ts_inputFile, "-o", _ts_outputFolder,
                    "--device", "cpu", "--ml", "--task", {safe_task}, "--fast"]

        # Run as subprocess in new process group (resource_tracker hang workaround)
        _ts_proc = _ts_subprocess.Popen(
            _ts_cmd, stdout=_ts_subprocess.DEVNULL, stderr=_ts_subprocess.PIPE,
            start_new_session=True,
        )
        _ts_timeout = {timeout_s}
        _ts_poll_interval = 5
        _ts_elapsed = 0
        _ts_prev_size = 0

        while _ts_elapsed < _ts_timeout:
            _ts_ret = _ts_proc.poll()
            if _ts_ret is not None:
                if _ts_ret != 0:
                    _ts_err = _ts_proc.stderr.read(8192).decode(errors='replace')[-500:]
                    raise RuntimeError(
                        f"TotalSegmentator exited with code {{_ts_ret}}: {{_ts_err}}"
                    )
                break
            if _ts_os.path.exists(_ts_outputFile):
                _ts_curr_size = _ts_os.path.getsize(_ts_outputFile)
                if _ts_curr_size > 1000 and _ts_curr_size == _ts_prev_size:
                    _ts_time.sleep(3)
                    break
                _ts_prev_size = _ts_curr_size
            _ts_time.sleep(_ts_poll_interval)
            _ts_elapsed += _ts_poll_interval

        # Kill process group if still running (resource_tracker hang)
{
        _kill_process_group_code(
            proc_var="_ts_proc",
            os_mod="_ts_os",
            signal_mod="_ts_signal",
            time_mod="_ts_time",
            logging_alias="_ts_logging",
            indent="        ",
        )
    }

        if not _ts_os.path.exists(_ts_outputFile):
            raise RuntimeError("TotalSegmentator did not produce output within timeout")

        # Import segmentation result
        _ts_logic = _ts_Logic()
        _ts_logic.readSegmentation({seg_var}, _ts_outputFile, {safe_task})

        {seg_var}.SetNodeReferenceID(
            {seg_var}.GetReferenceImageGeometryReferenceRole(), {volume_var}.GetID())
        {seg_var}.SetReferenceImageGeometryParameterFromVolumeNode({volume_var})

    except Exception as _ts_e:
        slicer.mrmlScene.RemoveNode({seg_var})
        raise ValueError(
            f"TotalSegmentator failed ({{type(_ts_e).__name__}}): {{_ts_e}}"
        ) from _ts_e
    finally:
{
        _kill_process_group_code(
            proc_var="_ts_proc",
            os_mod="_ts_os",
            signal_mod="_ts_signal",
            time_mod="_ts_time",
            logging_alias="_ts_logging",
            indent="        ",
        )
    }
        def _ts_rmtree_onerror(func, path, exc_info):
            import logging as _ts_log
            _ts_log.getLogger("slicer-mcp").warning(
                f"Failed to clean up temp file {{path}}: {{exc_info[1]}}"
            )

        if _ts_tempFolder is not None and _ts_os.path.isdir(_ts_tempFolder):
            _ts_shutil.rmtree(_ts_tempFolder, onerror=_ts_rmtree_onerror)
"""
