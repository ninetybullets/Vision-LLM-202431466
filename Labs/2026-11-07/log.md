ython Project03/bench.py fps --seconds 20 2>&1 | tee ../e1_bench_fps.log
[ WARN:0@2.808] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=1, value=3, duration=-1
Loading /home/ninetybullets/vision-llm/src/models/YOLO/yolo11n_int8.engine for TensorRT inference...
[11/08/2026-12:27:34] [TRT] [I] Loaded engine size: 5 MiB
[11/08/2026-12:27:34] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[11/08/2026-12:27:34] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 10 (MiB)
Loading /home/ninetybullets/vision-llm/src/models/YOLO/yolo11n_int8.engine for TensorRT inference...
[11/08/2026-12:27:35] [TRT] [I] Loaded engine size: 5 MiB
[11/08/2026-12:27:35] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[11/08/2026-12:27:35] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 10 (MiB)
[bench] 단독 구간 20.0초 측정 중
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
[reporter] 워밍업 5.7초
[bench] Gemma 로드: 73.37초
[bench] 동시 구간 20.0초 측정 중(Gemma 생성 중)
/tmp/pip-install-7l7pf0mj/llama-cpp-python_30c7f38315a7490fb2484cf4ae4d8096/vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:109: CUDA error
[New LWP 10976]
[New LWP 10977]
[New LWP 10980]
[New LWP 10987]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".
__futex_abstimed_wait_common64 (private=<optimized out>, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0xfffef4001e90) at ./nptl/futex-internal.c:57
57	./nptl/futex-internal.c: No such file or directory.
#0  __futex_abstimed_wait_common64 (private=<optimized out>, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0xfffef4001e90) at ./nptl/futex-internal.c:57
#1  __futex_abstimed_wait_common (cancel=true, private=<optimized out>, abstime=0x0, clockid=0, expected=0, futex_word=0xfffef4001e90) at ./nptl/futex-internal.c:87
#2  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0xfffef4001e90, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x0, private=<optimized out>) at ./nptl/futex-internal.c:139
#3  0x0000ffff87938834 in do_futex_wait (sem=sem@entry=0xfffef4001e90, abstime=0x0, clockid=0) at ./nptl/sem_waitcommon.c:111
#4  0x0000ffff879388f4 in __new_sem_wait_slow64 (sem=0xfffef4001e90, abstime=0x0, clockid=0) at ./nptl/sem_waitcommon.c:183
#5  0x0000aaaab5cf0f28 in PyThread_acquire_lock_timed ()
#6  0x0000aaaab5d4c40c in  ()
#7  0x0000aaaab5d351ac in  ()
#8  0x0000aaaab5d1c4dc in _PyEval_EvalFrameDefault ()
#9  0x0000aaaab5d34008 in _PyFunction_Vectorcall ()
#10 0x0000aaaab5d1c4dc in _PyEval_EvalFrameDefault ()
#11 0x0000aaaab5d34008 in _PyFunction_Vectorcall ()
#12 0x0000aaaab5d1c4dc in _PyEval_EvalFrameDefault ()
#13 0x0000aaaab5d34008 in _PyFunction_Vectorcall ()
#14 0x0000aaaab5d1c4dc in _PyEval_EvalFrameDefault ()
#15 0x0000aaaab5d34008 in _PyFunction_Vectorcall ()
#16 0x0000aaaab5d1c3a0 in _PyEval_EvalFrameDefault ()
#17 0x0000aaaab5d34008 in _PyFunction_Vectorcall ()
#18 0x0000aaaab5d1c3a0 in _PyEval_EvalFrameDefault ()
#19 0x0000aaaab5d34008 in _PyFunction_Vectorcall ()
#20 0x0000aaaab5d1c3a0 in _PyEval_EvalFrameDefault ()
#21 0x0000aaaab5e189f0 in  ()
#22 0x0000aaaab5e18874 in PyEval_EvalCode ()
#23 0x0000aaaab5e4cdec in  ()
#24 0x0000aaaab5e45318 in  ()
#25 0x0000aaaab5e4ca9c in  ()
#26 0x0000aaaab5e4bb34 in _PyRun_SimpleFileObject ()
#27 0x0000aaaab5e4b700 in _PyRun_AnyFileObject ()
#28 0x0000aaaab5e3bf00 in Py_RunMain ()
#29 0x0000aaaab5e08a08 in Py_BytesMain ()
#30 0x0000ffff878d7400 in __libc_start_call_main (main=main@entry=0xaaaab5e089e0, argc=argc@entry=5, argv=argv@entry=0xfffffdd351f8) at ../sysdeps/nptl/libc_start_call_main.h:58
#31 0x0000ffff878d74d8 in __libc_start_main_impl (main=0xaaaab5e089e0, argc=5, argv=0xfffffdd351f8, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=<optimized out>) at ../csu/libc-start.c:392
#32 0x0000aaaab5e088f0 in _start ()
[Inferior 1 (process 10968) detached]
(.venv) ninetybullets@ninetybullets-desktop:~/vision-llm$ CUDA_LAUNCH_BLOCKING=1 python Project03/main.py 2>&1 | tee ../e2_blocking.log
Event Log: /home/ninetybullets/vision-llm/Project03/logs/events-20261108-122948.jsonl
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
[reporter] 워밍업 1.8초
Gemma 로드: 47.14초
방송 문구 합성: 7.36초
[ WARN:0@68.692] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=1, value=3, duration=-1
Loading /home/ninetybullets/vision-llm/src/models/YOLO/yolo11n_int8.engine for TensorRT inference...
[11/08/2026-12:30:47] [TRT] [I] Loaded engine size: 5 MiB
[11/08/2026-12:30:48] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[11/08/2026-12:30:48] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 10 (MiB)
Loading /home/ninetybullets/vision-llm/src/models/YOLO/yolo11n_int8.engine for TensorRT inference...
[11/08/2026-12:30:49] [TRT] [I] Loaded engine size: 5 MiB
[11/08/2026-12:30:49] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[11/08/2026-12:30:49] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 10 (MiB)
Gtk-Message: 12:30:55.115: Failed to load module "canberra-gtk-module"
{"timestamp": 69.43171183899995, "target_id": 1, "kind": "appeared", "before": null, "after": {"zone": "near", "held_objects": []}}
/tmp/pip-install-7l7pf0mj/llama-cpp-python_30c7f38315a7490fb2484cf4ae4d8096/vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:109: CUDA error
[New LWP 11018]
[New LWP 11036]
[New LWP 11055]
[New LWP 11058]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".
__futex_abstimed_wait_common64 (private=<optimized out>, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0xfffed0000b70) at ./nptl/futex-internal.c:57
57	./nptl/futex-internal.c: No such file or directory.
#0  __futex_abstimed_wait_common64 (private=<optimized out>, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0xfffed0000b70) at ./nptl/futex-internal.c:57
#1  __futex_abstimed_wait_common (cancel=true, private=<optimized out>, abstime=0x0, clockid=0, expected=0, futex_word=0xfffed0000b70) at ./nptl/futex-internal.c:87
#2  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0xfffed0000b70, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x0, private=<optimized out>) at ./nptl/futex-internal.c:139
#3  0x0000ffff8bd48834 in do_futex_wait (sem=sem@entry=0xfffed0000b70, abstime=0x0, clockid=0) at ./nptl/sem_waitcommon.c:111
#4  0x0000ffff8bd488f4 in __new_sem_wait_slow64 (sem=0xfffed0000b70, abstime=0x0, clockid=0) at ./nptl/sem_waitcommon.c:183
#5  0x0000aaaae2740f28 in PyThread_acquire_lock_timed ()
#6  0x0000aaaae279c40c in  ()
#7  0x0000aaaae27851ac in  ()
#8  0x0000aaaae276c4dc in _PyEval_EvalFrameDefault ()
#9  0x0000aaaae2784008 in _PyFunction_Vectorcall ()
#10 0x0000aaaae276c4dc in _PyEval_EvalFrameDefault ()
#11 0x0000aaaae2784008 in _PyFunction_Vectorcall ()
#12 0x0000aaaae276c4dc in _PyEval_EvalFrameDefault ()
#13 0x0000aaaae2784008 in _PyFunction_Vectorcall ()
#14 0x0000aaaae276c4dc in _PyEval_EvalFrameDefault ()
#15 0x0000aaaae2784008 in _PyFunction_Vectorcall ()
#16 0x0000aaaae276c3a0 in _PyEval_EvalFrameDefault ()
#17 0x0000aaaae2784008 in _PyFunction_Vectorcall ()
#18 0x0000aaaae276c3a0 in _PyEval_EvalFrameDefault ()
#19 0x0000aaaae2784008 in _PyFunction_Vectorcall ()
#20 0x0000aaaae276c3a0 in _PyEval_EvalFrameDefault ()
#21 0x0000aaaae28689f0 in  ()
#22 0x0000aaaae2868874 in PyEval_EvalCode ()
#23 0x0000aaaae289cdec in  ()
#24 0x0000aaaae2895318 in  ()
#25 0x0000aaaae289ca9c in  ()
#26 0x0000aaaae289bb34 in _PyRun_SimpleFileObject ()
#27 0x0000aaaae289b700 in _PyRun_AnyFileObject ()
#28 0x0000aaaae288bf00 in Py_RunMain ()
#29 0x0000aaaae2858a08 in Py_BytesMain ()
#30 0x0000ffff8bce7400 in __libc_start_call_main (main=main@entry=0xaaaae28589e0, argc=argc@entry=2, argv=argv@entry=0xffffdfcdb778) at ../sysdeps/nptl/libc_start_call_main.h:58
#31 0x0000ffff8bce74d8 in __libc_start_main_impl (main=0xaaaae28589e0, argc=2, argv=0xffffdfcdb778, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=<optimized out>) at ../csu/libc-start.c:392
#32 0x0000aaaae28588f0 in _start ()
[Inferior 1 (process 11006) detached]
