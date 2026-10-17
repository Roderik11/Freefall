# meshoptimizer

Mesh simplification for the LODs the model importer generates (`Freefall.Engine/Assets/Importers/LODGenerator.cs`).

- Source: https://github.com/zeux/meshoptimizer, release **v1.3** (2026-09-25), `meshoptimizer-1.3.tar.gz`
  (sha256 `d50a0b5e1323ba4379cdb32f89436c20652b7393445f8e18c88ae64445b67b77`). Only the library (`src/`) is kept.
- Licence: MIT, see `LICENSE`.
- `win-x64/meshoptimizer.dll` is built from `src/` by `build.cmd` (MSVC x64, static CRT, needs only KERNEL32) and
  committed. `Freefall.Engine.csproj` copies it next to the engine, as PhysX's native DLLs are.
- The engine calls it by P/Invoke: `Freefall.Engine/Graphics/MeshOptimizer.cs`. When upgrading, replace `src/`, run
  `build.cmd`, and check the signatures there against `src/meshoptimizer.h`.
