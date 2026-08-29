# Edge AI Runtime — Agent Doctrine

## What this repo is
Edge-native AI runtime: inference at the edge, WebAssembly model execution, ONNX, and sub-50ms response patterns.
By Cherry Shanaley (Chan), AI Solutions Engineer.

## Tech stack
- TypeScript strict, Vercel Edge Runtime, WebAssembly (WASM), ONNX Runtime Web, Cloudflare Workers

## Coding rules
- All edge functions exported from `runtime/` — no Node.js APIs
- WASM modules in `wasm/` — loaded via `WebAssembly.instantiateStreaming` for streaming init
- ONNX models in `models/` — quantized INT8 only for edge (size constraint)
- Inference pipelines in `lib/inference/` — typed input/output tensors
- Response time target: < 50ms TTFT at edge
- Model size limit: 25MB per model (Vercel Edge constraint)

## Commands
```bash
npm install && npm run dev
npm run build:wasm
npm run bench:edge
npm run test
```

## Do not
- Import `fs`, `path`, `crypto` (Node built-ins) in edge functions
- Use unquantized models — INT8 only
- Exceed 25MB model size
