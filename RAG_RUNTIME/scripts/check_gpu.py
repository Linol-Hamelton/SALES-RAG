#!/usr/bin/env python3
"""Check GPU/CUDA environment and validate key model dependencies."""
import sys
import platform
import subprocess
from pathlib import Path

def check_python():
    version = sys.version_info
    ok = version >= (3, 11)
    print(f"{'✓' if ok else '✗'} Python {version.major}.{version.minor}.{version.micro}", end="")
    if not ok:
        print(" (need 3.11+)")
    else:
        print()
    return ok

def check_torch():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✓ PyTorch {torch.__version__}")
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {name} ({vram:.1f} GB VRAM)")
        else:
            print("  No CUDA GPU detected (CPU mode)")
        return True, cuda_available
    except ImportError:
        print("✗ PyTorch not installed")
        return False, False

def check_qdrant():
    try:
        from qdrant_client import QdrantClient
        import importlib.metadata
        version = importlib.metadata.version("qdrant-client")
        print(f"✓ qdrant-client {version}")
        return True
    except Exception:
        print("✗ qdrant-client not installed")
        return False

def check_sentence_transformers():
    try:
        import sentence_transformers
        print(f"✓ sentence-transformers {sentence_transformers.__version__}")
        return True
    except ImportError:
        print("✗ sentence-transformers not installed")
        return False

def check_openai():
    try:
        import openai
        print(f"✓ openai {openai.__version__}")
        return True
    except ImportError:
        print("✗ openai not installed")
        return False

def test_embedding(cuda_available: bool):
    """Try loading BGE-M3 and encoding a test sentence."""
    model_path = Path(__file__).parent.parent / "models" / "embeddings" / "BAAI_bge-m3"
    if not model_path.exists():
        print(f"  BGE-M3 not downloaded yet at {model_path}")
        print(f"  Run: huggingface-cli download BAAI/bge-m3 --local-dir {model_path}")
        return False

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if cuda_available else "cpu"
        print(f"  Loading BGE-M3 on {device}...")
        model = SentenceTransformer(str(model_path), device=device)

        test_text = "световая вывеска кофейня монтаж и сборка"
        embedding = model.encode([test_text], normalize_embeddings=True)
        print(f"  ✓ BGE-M3 loaded! Embedding shape: {embedding.shape}, device: {device}")

        if cuda_available:
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM used: {vram_used:.2f} GB")

        del model
        if cuda_available:
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ✗ BGE-M3 load failed: {e}")
        return False

def main():
    print("=" * 60)
    print("SALES_RAG Environment Check")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("=" * 60)

    checks = []
    checks.append(check_python())
    torch_ok, cuda_available = check_torch()
    checks.append(torch_ok)
    checks.append(check_sentence_transformers())
    checks.append(check_qdrant())
    checks.append(check_openai())

    print("\nModel test:")
    test_embedding(cuda_available)

    print("\n" + "=" * 60)
    all_ok = all(checks)
    if all_ok:
        print("✓ Environment ready!")
    else:
        print("✗ Some checks failed. Install missing packages.")
        sys.exit(1)

if __name__ == "__main__":
    main()
