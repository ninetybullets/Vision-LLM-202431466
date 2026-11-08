Event Log: /home/ninetybullets/vision-llm/Project03/logs/events-20261108-121448.jsonl
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
[reporter] 워밍업 1.4초
Gemma 로드: 41.58초
방송 문구 합성: 7.21초
[ WARN:0@58.337] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=1, value=3, duration=-1
Loading /home/ninetybullets/vision-llm/src/models/YOLO/yolo11n_int8.engine for TensorRT inference...
[11/08/2026-12:15:40] [TRT] [I] Loaded engine size: 5 MiB
[11/08/2026-12:15:40] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[11/08/2026-12:15:40] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 10 (MiB)
Loading /home/ninetybullets/vision-llm/src/models/YOLO/yolo11n_int8.engine for TensorRT inference...
[11/08/2026-12:15:41] [TRT] [I] Loaded engine size: 5 MiB
[11/08/2026-12:15:41] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[11/08/2026-12:15:41] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 10 (MiB)
Gtk-Message: 12:15:44.999: Failed to load module "canberra-gtk-module"
{"timestamp": 58.92334201299991, "target_id": 2, "kind": "appeared", "before": null, "after": {"zone": "near", "held_objects": []}}
/tmp/pip-install-7l7pf0mj/llama-cpp-python_30c7f38315a7490fb2484cf4ae4d8096/vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:109: CUDA error
[New LWP 9625]
[New LWP 9627]
[New LWP 9646]
[New LWP 9648]
[New LWP 9649]
[New LWP 9650]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".
__futex_abstimed_wait_common64 (private=<optimized out>, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0xfffed4000b70) at ./nptl/futex-internal.c:57
57	./nptl/futex-internal.c: No such file or directory.
#0  __futex_abstimed_wait_common64 (private=<optimized out>, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0xfffed4000b70) at ./nptl/futex-internal.c:57
#1  __futex_abstimed_wait_common (cancel=true, private=<optimized out>, abstime=0x0, clockid=0, expected=0, futex_word=0xfffed4000b70) at ./nptl/futex-internal.c:87
#2  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0xfffed4000b70, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x0, private=<optimized out>) at ./nptl/futex-internal.c:139
#3  0x0000ffff8f0f8834 in do_futex_wait (sem=sem@entry=0xfffed4000b70, abstime=0x0, clockid=0) at ./nptl/sem_waitcommon.c:111
#4  0x0000ffff8f0f88f4 in __new_sem_wait_slow64 (sem=0xfffed4000b70, abstime=0x0, clockid=0) at ./nptl/sem_waitcommon.c:183
#5  0x0000aaaadf230f28 in PyThread_acquire_lock_timed ()
#6  0x0000aaaadf28c40c in  ()
#7  0x0000aaaadf2751ac in  ()
#8  0x0000aaaadf25c4dc in _PyEval_EvalFrameDefault ()
#9  0x0000aaaadf274008 in _PyFunction_Vectorcall ()
#10 0x0000aaaadf25c4dc in _PyEval_EvalFrameDefault ()
#11 0x0000aaaadf274008 in _PyFunction_Vectorcall ()
#12 0x0000aaaadf25c4dc in _PyEval_EvalFrameDefault ()
#13 0x0000aaaadf274008 in _PyFunction_Vectorcall ()
#14 0x0000aaaadf25c4dc in _PyEval_EvalFrameDefault ()
#15 0x0000aaaadf274008 in _PyFunction_Vectorcall ()
#16 0x0000aaaadf25c3a0 in _PyEval_EvalFrameDefault ()
#17 0x0000aaaadf274008 in _PyFunction_Vectorcall ()
#18 0x0000aaaadf25c3a0 in _PyEval_EvalFrameDefault ()
#19 0x0000aaaadf274008 in _PyFunction_Vectorcall ()
#20 0x0000aaaadf25c3a0 in _PyEval_EvalFrameDefault ()
#21 0x0000aaaadf3589f0 in  ()
#22 0x0000aaaadf358874 in PyEval_EvalCode ()
#23 0x0000aaaadf38cdec in  ()
#24 0x0000aaaadf385318 in  ()
#25 0x0000aaaadf38ca9c in  ()
#26 0x0000aaaadf38bb34 in _PyRun_SimpleFileObject ()
#27 0x0000aaaadf38b700 in _PyRun_AnyFileObject ()
#28 0x0000aaaadf37bf00 in Py_RunMain ()
#29 0x0000aaaadf348a08 in Py_BytesMain ()
#30 0x0000ffff8f097400 in __libc_start_call_main (main=main@entry=0xaaaadf3489e0, argc=argc@entry=2, argv=argv@entry=0xffffca6c3b58) at ../sysdeps/nptl/libc_start_call_main.h:58
#31 0x0000ffff8f0974d8 in __libc_start_main_impl (main=0xaaaadf3489e0, argc=2, argv=0xffffca6c3b58, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=<optimized out>) at ../csu/libc-start.c:392
#32 0x0000aaaadf3488f0 in _start ()
[Inferior 1 (process 9605) detached]
Aborted (core dumped)
