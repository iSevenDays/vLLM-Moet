# T7 — upstream recipe on sm_89: deepseek_mtp INCOMPATIBLE with the 0731 checkpoint

The upstream recipe (`pro6000x2-tp2.yaml`) uses `--speculative-config
{"method":"deepseek_mtp","num_speculative_tokens":2}`. On the 0731 checkpoint this
crashes during MTP drafter weight load:

    File ".../vllm/models/deepseek_v4/nvidia/mtp.py", line 459, in load_weights
        param = params_dict[name]
    KeyError: 'model.layers.43.mtp_block.main_norm.weight'

Root cause: 0731 uses **DSpark** MTP (`dspark_block_size 5`,
`dspark_target_layer_ids [40,41,42]`, `dspark_markov_rank 256` — README §0). Its
MTP-block weight names/structure do not match the standard `deepseek_mtp` loader.
So the speculation-method H4 variable (dspark k=5 vs deepseek_mtp k=2) is
**untestable on this checkpoint** — the method is checkpoint-bound to DSpark.

The other H4 variables (chunk 4096, ctx 131072, util 0.92, seqs 2) loaded fine
(moe_w2 exact cache 29.99 GiB OK); the crash was solely the drafter. To test the
remaining H4 variables, keep DSpark and vary K: `dspark` with
`num_speculative_tokens=2` + chunk 4096 + ctx 131072 + util 0.92.

(Boot also reduced --kv-cache-memory-bytes to 2 GiB for the chunk-4096 recompile
headroom; that part was fine.)
