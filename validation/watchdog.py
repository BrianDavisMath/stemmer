"""External macOS memory watchdog for explicitly requested local benchmarks.

This module never imports torch. Telemetry failure invalidates and stops a run.
Available memory is psutil's macOS estimate, not a promise of allocatable RAM.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3


def parse_vm_stat(output):
    page_size = int(re.search(r'page size of (\d+) bytes', output).group(1))
    counters = dict((key.strip().strip('"'), int(value)) for key, value in
                    re.findall(r'^([^:\n]+):\s+(\d+)\.', output, re.M))
    return dict(swapins=counters['Swapins'], swapouts=counters['Swapouts'],
                pageouts=counters['Pageouts'],
                compressed_bytes=counters['Pages occupied by compressor'] * page_size,
                page_size=page_size)


def sample_system():
    import psutil
    vm = subprocess.run(['/usr/bin/vm_stat'], capture_output=True, text=True,
                        check=True, timeout=1).stdout
    ctl = subprocess.run(['/usr/sbin/sysctl', 'vm.swapusage',
                          'kern.memorystatus_vm_pressure_level'],
                         capture_output=True, text=True, check=True, timeout=1).stdout
    swap = re.search(r'used = ([\d.]+)([KMG])', ctl)
    pressure = re.search(r'kern.memorystatus_vm_pressure_level: (\d+)', ctl)
    if not swap or not pressure:
        raise RuntimeError('Incomplete system swap/pressure telemetry')
    memory = psutil.virtual_memory()
    return dict(time=time.time(), available_bytes=memory.available,
                total_bytes=memory.total,
                swap_bytes=float(swap[1]) * 1024 ** ('KMG'.index(swap[2]) + 1),
                pressure=int(pressure[1]), **parse_vm_stat(vm))


def violation(sample, baseline, reserve_bytes, rss_limit_bytes, pageout_policy="abort"):
    if pageout_policy not in ("abort", "observe"):
        raise ValueError("Unknown pageout policy")
    if sample['swap_bytes'] > baseline['swap_bytes']:
        return 'system swap usage grew'
    for key in ('swapins', 'swapouts') + (('pageouts',) if pageout_policy == 'abort' else ()):
        if sample[key] > baseline[key]:
            return f'system {key} grew'
    if sample['pressure'] != 1:
        return 'system memory pressure is not normal'
    if sample['available_bytes'] < reserve_bytes:
        return 'reserved available-memory headroom crossed'
    if sample['compressed_bytes'] - baseline['compressed_bytes'] > GIB:
        return 'system compressor grew by more than 1 GiB'
    if sample.get('rss_bytes', 0) > rss_limit_bytes:
        return 'benchmark process-tree RSS limit crossed'
    return None


def stop_group(child):
    # A new session isolates this benchmark and all of its descendants.
    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can report EPERM after the last group member exits. Never
            # treat EPERM as success unless an independent census finds no member.
            import psutil
            for process in psutil.process_iter():
                try:
                    if os.getpgid(process.pid) == child.pid:
                        raise
                except ProcessLookupError:
                    pass
    send(signal.SIGTERM)
    try:
        child.wait(timeout=.5)
    except subprocess.TimeoutExpired:
        pass
    send(signal.SIGKILL)
    child.wait(timeout=5)


def supervise(command, log_path, *, reserve_gib=10, max_rss_gib=24,
              deadline=900, interval=.25, sampler=sample_system, status_file=None,
              rss_budget_policy="baseline", pageout_policy="abort"):
    import psutil
    if any(not math.isfinite(value) or value <= 0
           for value in (reserve_gib, max_rss_gib, deadline, interval)):
        raise ValueError('watchdog budgets, deadline and interval must be finite and positive')
    if rss_budget_policy not in ('baseline', 'live-reserve'):
        raise ValueError('Unknown RSS budget policy')
    if pageout_policy not in ("abort", "observe"):
        raise ValueError("Unknown pageout policy")
    child = None
    reason = None
    baseline = None
    last = None
    peak_rss = 0
    minimum_available = None
    stage_peaks = {}
    stage = 'starting'
    previous_sample_time = None
    maximum_sample_gap = 0
    started_wall = time.time()
    started = time.monotonic()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x') as log:
        def record(event, **values):
            log.write(json.dumps(dict(event=event, **values)) + '\n')
            log.flush()

        try:
            baseline = sampler()
            previous_sample_time = baseline.get('time', time.time())
            last = baseline
            minimum_available = baseline['available_bytes']
            reserve = reserve_gib * GIB
            # Budget responds to current availability; it never expands mid-run.
            rss_limit = max_rss_gib * GIB
            if rss_budget_policy == 'baseline':
                rss_limit = min(rss_limit, baseline['available_bytes'] - reserve)
            # Explicit experiments may use the requested cap while still
            # enforcing the live system reserve on every sample.
            record('baseline', **baseline, reserve_bytes=reserve, rss_limit_bytes=rss_limit,
                   rss_budget_policy=rss_budget_policy, pageout_policy=pageout_policy)
            reason = violation(baseline, baseline, reserve, rss_limit, pageout_policy)
            if baseline['swap_bytes'] != 0:
                reason = 'clean zero-swap baseline required'
            if rss_limit <= 0:
                reason = 'no memory budget remains after reserving headroom'
            if not reason:
                env = dict(os.environ, KARAOKE_BENCH_GUARD_PID=str(os.getpid()))
                child = subprocess.Popen(command, env=env, start_new_session=True)
                record('child', pid=child.pid, command=command)
                while True:
                    last = sampler()
                    sample_time = last.get('time', time.time())
                    gap = sample_time - previous_sample_time
                    previous_sample_time = sample_time
                    maximum_sample_gap = max(maximum_sample_gap, gap)
                    continuity_failure = ('system telemetry gap or clock discontinuity'
                                          if gap > max(5, interval * 4) or gap < -1 else None)
                    try:
                        root = psutil.Process(child.pid)
                        processes = [root, *root.children(recursive=True)]
                        rss = 0
                        for process in processes:
                            try:
                                rss += process.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        rss = 0
                    last['rss_bytes'] = rss
                    if status_file is not None:
                        try:
                            stage = json.loads(Path(status_file).read_text())['stage']
                        except (OSError, ValueError, KeyError):
                            pass
                        last['stage'] = stage
                        stage_peaks[stage] = max(stage_peaks.get(stage, 0), rss)
                    peak_rss = max(peak_rss, rss)
                    minimum_available = min(minimum_available, last['available_bytes'])
                    record('sample', **last)
                    reason = continuity_failure or violation(last, baseline, reserve, rss_limit, pageout_policy)
                    if max(time.monotonic() - started, time.time() - started_wall) > deadline:
                        reason = 'watchdog wall-clock deadline exceeded'
                    if reason or child.poll() is not None:
                        break
                    time.sleep(interval)
        except Exception as error:
            reason = f'telemetry or supervisor failure: {type(error).__name__}: {error}'
        finally:
            if child is not None:
                try:
                    stop_group(child)
                except Exception as error:
                    reason = f'process cleanup failed: {type(error).__name__}: {error}'
            try:
                final = sampler()
                record('after', **final)
                if previous_sample_time is not None:
                    gap = final.get('time', time.time()) - previous_sample_time
                    maximum_sample_gap = max(maximum_sample_gap, gap)
                    if gap > max(5, interval * 4) or gap < -1:
                        reason = reason or 'system telemetry gap or clock discontinuity'
                if baseline is not None:
                    minimum_available = min(minimum_available, final['available_bytes'])
                    reason = reason or violation(final, baseline, reserve, rss_limit, pageout_policy)
            except Exception as error:
                reason = reason or f'final telemetry unavailable: {error}'
            code = child.returncode if child else None
            valid = reason is None and code == 0
            summary = dict(event='guard_result', valid=valid, reason=reason,
                           child_exit_code=code, peak_tree_rss_gib=peak_rss / GIB,
                           minimum_available_gib=minimum_available / GIB if minimum_available is not None else None,
                           stage_peak_tree_rss_gib={key: value / GIB for key, value in stage_peaks.items()},
                           maximum_sample_gap_seconds=maximum_sample_gap,
                           seconds=time.monotonic() - started, log=str(log_path))
            record('result', **{k: v for k, v in summary.items() if k != 'event'})
            print(json.dumps(summary), flush=True)
    return 0 if valid else 1


def guarded_entry(main):
    """Re-exec under an independent watchdog before any model imports."""
    if os.environ.get('KARAOKE_BENCH_GUARD_PID') == str(os.getppid()):
        main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-log', type=Path,
                        default=Path('.runtime') / f'scnet-guard-{time.time_ns()}.jsonl')
    parser.add_argument('--guard-reserve-gib', type=float, default=10)
    parser.add_argument('--guard-max-rss-gib', type=float, default=24)
    parser.add_argument('--guard-rss-budget-policy', choices=('baseline', 'live-reserve'), default='baseline')
    parser.add_argument('--guard-pageout-policy', choices=('abort', 'observe'), default='abort')
    parser.add_argument('--guard-deadline', type=float, default=900)
    parser.add_argument('--guard-status-file', type=Path)
    args, remaining = parser.parse_known_args()
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.guard_reserve_gib, args.guard_max_rss_gib, args.guard_deadline)):
        parser.error('guard budgets and deadline must be finite and positive')
    # Help does not import torch or require system telemetry.
    if '--help' in remaining or '-h' in remaining:
        sys.argv = [sys.argv[0], *remaining]
        main()
        return
    def interrupted(signum, frame):
        raise InterruptedError(f'watchdog interrupted by signal {signum}')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = supervise([sys.executable, str(Path(sys.argv[0]).resolve()), *remaining],
                         args.guard_log, reserve_gib=args.guard_reserve_gib,
                         max_rss_gib=args.guard_max_rss_gib,
                         deadline=args.guard_deadline, status_file=args.guard_status_file,
                         rss_budget_policy=args.guard_rss_budget_policy,
                         pageout_policy=args.guard_pageout_policy)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    raise SystemExit(code)
