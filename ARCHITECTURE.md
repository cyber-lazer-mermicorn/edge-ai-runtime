# Architecture

## Overview
Edge AI Runtime executes quantized ONNX models at the edge (Vercel / Cloudflare) with WASM acceleration for sub-50ms inference.

## Layers

| Layer | Path | Responsibility |
|---|---|---|
| Edge Functions | `runtime/` | Edge-compatible inference endpoints |
| WASM Modules | `wasm/` | Compiled acceleration modules |
| ONNX Models | `models/` | INT8 quantized models (<25MB) |
| Inference | `lib/inference/` | Typed tensor pipeline |
| Benchmarks | `benchmarks/` | TTFT and throughput at edge |

## Model lifecycle
Train/convert → quantize to INT8 → validate accuracy loss < 2% → add to `models/` → register in `lib/inference/registry.ts`.
