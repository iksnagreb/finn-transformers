language batch size 8 einbruch
    - num threads für computer erhöhen auf 16 (vorher: 8, nachher 16 num threads bei sessoptions):
        - vorher bei batch size 8: 7600 batches/s
        - nachher bei batch size 8: 7600 batches/s
        - vorher bei batch size 16: 2500 batches/s
        - nachher bei batch size 16: 4000 batches/S

        --> weniger Einbruch an throughput, aber immer noch da...
        --> kernel=32 genauso wie kernels=8

        - auf dem Jetson: für 1...13 Threads getestet, kaum unterschiede



        if "CUDAExecutionProvider" in available:
        cuda_providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": GPU_MEM_LIMIT_BYTES,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",    # EXHAUSTIVE for best performance, DEFAULT
                    "do_copy_in_default_stream": True,
                },
            ),
            ("CPUExecutionProvider", {}),
        ]



# Einstellung im measure skript für output größen max. größe (16MB, bei batch size 8)
# test auf dem Jetson mit dem Language modell