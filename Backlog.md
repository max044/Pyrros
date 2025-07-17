# 🧠 Pyrros – Backlog

---

## 🔜 TODO

- [ ] **Implement `trainer/sft.py`**
- [ ] **Build unit tests**
- [ ] **Verify all features**
      LoRA, gradient accumulation, eval, early stopping, etc.

- [ ] **Multi-GPU support** (`FSDP`, `DDP`)
      Use Composer's distributed engine with a simple interface (`launch(config)` or CLI flags).

- [ ] **Improve `load_ref_model`**
      Handle LoRA/full model cases. Ensure compatibility with FSDP and safe eval mode.

- [ ] **Benchmark against Hugging Face and unsloth Trainers**
      Compare speed, memory, accuracy on GSM8K. Log results for blog & README.


- [ ] **Generate Qwen3 recipes for some config**
      1.5B → 4B → 7B with presets: 1×A100, 2×A100, etc.

---

## 🟡 DOING

- [ ] **Clean and comment code**
- [ ] **GRPO recipe**
      GRPO + QLoRA on Qwen. Work in progress: cleanup, eval, reward config.
- [ ] **Interactive CLI (arrow-based UX)**
      Shadcn like

---

## ✅ DONE

- [x] **Loggers**
- [x] **Basic CLI**
- [x] **GRPO trainer implementation**
- [x] **Test on Beam Cloud**
- [x] **Smoke test**
- [x] **Project structure**

---

## 🧭 SOMEDAY

- [ ] **Implement PPO**
- [ ] **Implement DPO**
- [ ] **Implement SEAL**
- [ ] **Triton kernel patching**

- [ ] **vLLM generation backend grpo_sampler**

- [ ] **Website (landing + doc + recipes list)**
      Static landing + Mintlify doc + links to benchmarks.

- [ ] **Blogposts per recipe**
      Benchmarks + method explanation. Start with GRPO/Qwen.

- [ ] **Mintlify documentation**
      Public docs with API, module structure, and usage guides.

- [ ] **Automated test suite (Beam Cloud)**
- [ ] **Courses / e-learning**
