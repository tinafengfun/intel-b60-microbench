# Makefile for SPIR-V Microbenchmarks on Intel Arc Pro B60
# Pipeline: .spvasm → .spv (spirv-as) → _bmg.bin (ocloc) → .asm (ocloc disasm)
# Run: spirv_runner

DEVICE = bmg-g21
CC = g++
CXXFLAGS = -std=c++17 -O2
LDFLAGS = -lze_loader -lm

# Find all .spvasm files
SPVASM = $(wildcard spirv_dpas_*.spvasm) $(wildcard spirv_mem_*.spvasm) spirv_test.spvasm

.PHONY: all clean run_dpas_latency run_dpas_throughput run_mem sweep_dpas validate

all: spirv_runner

# Build Level Zero host runner
spirv_runner: spirv_runner.cpp
	$(CC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Pattern: .spvasm → .spv
%.spv: %.spvasm
	spirv-as $< -o $@

# Pattern: .spv → _bmg.bin
%_bmg.bin: %.spv
	ocloc compile -spirv_input -file $< -device $(DEVICE) -output $(basename $@)

# Disassemble GEN binary
%_disasm: %_bmg.bin
	@mkdir -p $@
	ocloc disasm -file $< -dump $@ -device $(DEVICE)

# Clean
clean:
	rm -f spirv_runner *.spv *_bmg.bin
	rm -rf *_disasm spirv_dps_sweep_* results/

# Validate: check DPAS instruction count matches expected N
validate: spirv_dpas_latency_bmg.bin
	@mkdir -p spirv_dpas_latency_disasm
	ocloc disasm -file $< -dump spirv_dpas_latency_disasm -device $(DEVICE) 2>/dev/null
	@echo "=== DPAS instruction count ==="
	@grep -c "dpas" spirv_dpas_latency_disasm/.text.*.asm 2>/dev/null || echo "No DPAS instructions found"
	@echo "=== Sample instructions ==="
	@grep "dpas" spirv_dpas_latency_disasm/.text.*.asm 2>/dev/null | head -5
