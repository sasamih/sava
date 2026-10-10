# Hand-written SAVA kernel: y = relu(x) * 2 over 8192 f32 elements.
#
# The vector is processed in chunks of 1024 elements (4 KiB):
#   DRAM --dma.ld--> SRAM[0x0000] --v.relu/v.muls--> SRAM[0x1000] --dma.st--> DRAM
#
# Every cross-unit dependency needs an explicit `wait`; delete one and the
# simulator reports a hazard. This version is fully serialized -- making it
# double-buffered (overlapping the next load with the current compute) is a
# good first exercise (see docs/roadmap.md).
#
#   python -m sava.asm examples/asm/relu_scale.s -o build/relu_scale.sbin
#   ./build/sim/sava-sim build/relu_scale.sbin -i x.npy -o y.npy --stats

.kernel relu_scale
.arg x in f32 8192
.arg y out f32 8192

    csrwi DMA_ROWS, 1
    csrwi DMA_ROW_BYTES, 4096       # one 1024 x f32 chunk per transfer
    csrwi VPU_LEN, 1024
    li x10, 0                       # byte offset of the current chunk
    li x11, 32768                   # total bytes
    li x12, 0x0000                  # SRAM input chunk
    li x13, 0x1000                  # SRAM output chunk
loop:
    add x14, a0, x10
    dma.ld x12, x14, 0              # SRAM[x12] <- x[chunk]          (tag 0)
    wait dma0                       # RAW: VPU reads what the DMA writes
    v.relu x13, x12
    v.muls x13, x13, 2.0
    wait vpu                        # RAW: DMA reads what the VPU writes
    add x15, a1, x10
    dma.st x13, x15, 1              # y[chunk] <- SRAM[x13]          (tag 1)
    wait dma1                       # WAR: next v.relu overwrites SRAM[x13]
    addi x10, x10, 4096
    blt x10, x11, loop
    halt
