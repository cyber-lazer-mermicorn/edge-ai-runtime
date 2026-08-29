# Copilot Instructions — Edge AI Runtime

## Always
- Use only Web APIs in edge functions (no Node.js built-ins)
- Load WASM via `WebAssembly.instantiateStreaming`
- Use INT8 quantized ONNX models
- Type all tensor inputs/outputs explicitly
- Target < 50ms TTFT

## Never
- Import `fs`, `path`, `os`, `crypto` in edge code
- Use unquantized (FP32) models at edge
- Exceed 25MB model size limit
