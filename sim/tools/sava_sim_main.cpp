// sava-sim: standalone runner for SAVA executables.
//
//   sava-sim kernel.sbin -i a.npy -i b.npy -o c.npy [--stats] [--trace[=FILE]]
//   sava-sim kernel.sbin --disasm
//
// Kernel arguments are bound in the order of the executable's arg table:
// every `in`/`inout` argument consumes the next -i file, every `out`/`inout`
// argument consumes the next -o file.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../src/device.h"
#include "../src/executable.h"
#include "npy.h"
#include "sava/sava_sim.h"

namespace {

const char* kUsage =
    "usage: sava-sim KERNEL.sbin [options]\n"
    "  -i FILE.npy             input for the next in/inout argument\n"
    "  -o FILE.npy             output for the next out/inout argument\n"
    "  --stats                 print performance counters\n"
    "  --trace[=FILE]          instruction trace (default: stderr)\n"
    "  --disasm                print arg table and disassembly, then exit\n"
    "  --no-strict             do not trap on unsynchronized accesses\n"
    "  --max-instructions=N    abort after N instructions\n";

const char* npyDescr(sava::DType dtype) {
  switch (dtype) {
    case sava::DType::F32: return "<f4";
    case sava::DType::BF16: return "<u2";
    case sava::DType::I32: return "<i4";
    case sava::DType::I8: return "|i1";
  }
  return "";
}

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "sava-sim: error: %s\n", msg.c_str());
  std::exit(1);
}

void printStats(const sava_stats& s) {
  auto pct = [&](uint64_t busy) { return s.cycles ? 100.0 * double(busy) / double(s.cycles) : 0.0; };
  const double us = double(s.cycles) / (sava::defaults::kClockGhz * 1e3);
  std::printf("cycles              %12" PRIu64 "   (%.2f us @ %.1f GHz)\n", s.cycles, us,
              sava::defaults::kClockGhz);
  std::printf("instructions        %12" PRIu64 "   (scalar %" PRIu64 ")\n", s.instructions,
              s.scalar_instructions);
  std::printf("wait stalls         %12" PRIu64 "\n", s.wait_stall_cycles);
  std::printf("queue-full stalls   %12" PRIu64 "\n", s.queue_stall_cycles);
  std::printf("dma  ops %8" PRIu64 "  busy %10" PRIu64 " (%5.1f%%)  in %" PRIu64 " B  out %" PRIu64 " B\n",
              s.dma_ops, s.dma_busy_cycles, pct(s.dma_busy_cycles), s.dma_bytes_in, s.dma_bytes_out);
  std::printf("mxu  ops %8" PRIu64 "  busy %10" PRIu64 " (%5.1f%%)  macs %" PRIu64 "\n", s.mxu_ops,
              s.mxu_busy_cycles, pct(s.mxu_busy_cycles), s.mxu_macs);
  std::printf("vpu  ops %8" PRIu64 "  busy %10" PRIu64 " (%5.1f%%)  elements %" PRIu64 "\n", s.vpu_ops,
              s.vpu_busy_cycles, pct(s.vpu_busy_cycles), s.vpu_elements);
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel_path, trace_path;
  std::vector<std::string> inputs, outputs;
  bool stats = false, disasm = false, trace = false, strict = true;
  uint64_t max_instructions = 0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-i" && i + 1 < argc) inputs.push_back(argv[++i]);
    else if (a == "-o" && i + 1 < argc) outputs.push_back(argv[++i]);
    else if (a == "--stats") stats = true;
    else if (a == "--disasm") disasm = true;
    else if (a == "--no-strict") strict = false;
    else if (a == "--trace") { trace = true; trace_path = "-"; }
    else if (a.rfind("--trace=", 0) == 0) { trace = true; trace_path = a.substr(8); }
    else if (a.rfind("--max-instructions=", 0) == 0) max_instructions = std::stoull(a.substr(19));
    else if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); return 0; }
    else if (!a.empty() && a[0] != '-' && kernel_path.empty()) kernel_path = a;
    else die("unknown argument '" + a + "'\n" + kUsage);
  }
  if (kernel_path.empty()) die(std::string("missing kernel\n") + kUsage);

  std::ifstream f(kernel_path, std::ios::binary);
  if (!f) die("cannot open " + kernel_path);
  std::vector<uint8_t> image((std::istreambuf_iterator<char>(f)), {});

  sava::Executable exe;
  std::string error;
  if (!sava::parseExecutable(image.data(), image.size(), &exe, &error)) die(error);

  if (disasm) {
    std::printf("kernel %s (sbin v%u.%u), %zu instructions\n", exe.name.c_str(),
                exe.version_major, exe.version_minor, exe.code.size());
    for (size_t i = 0; i < exe.args.size(); ++i) {
      const auto& arg = exe.args[i];
      std::string dims;
      for (size_t d = 0; d < arg.dims.size(); ++d) dims += (d ? "x" : "") + std::to_string(arg.dims[d]);
      std::printf("  a%zu: %-5s %-12s %s[%s]\n", i, sava::argKindName(arg.kind), arg.name.c_str(),
                  sava::dtypeName(arg.dtype), dims.c_str());
    }
    for (size_t pc = 0; pc < exe.code.size(); ++pc)
      std::printf("%6zu: %016" PRIx64 "  %s\n", pc, exe.code[pc],
                  sava::disassemble(exe.code[pc], pc).c_str());
    return 0;
  }

  sava_config config;
  sava_config_init_default(&config);
  config.strict_hazards = strict ? 1 : 0;
  config.max_instructions = max_instructions;
  config.trace_path = trace ? trace_path.c_str() : nullptr;
  sava_device* dev = sava_device_create(&config);
  if (!dev) die("cannot create device");
  auto check = [&](sava_status st) {
    if (st != SAVA_OK) die(sava_device_last_error(dev));
  };
  check(sava_load_executable(dev, image.data(), image.size()));

  // Simple bump allocator over device DRAM; address 0 stays unused.
  uint64_t next_addr = 0x1000;
  std::vector<uint64_t> args;
  size_t next_in = 0, next_out = 0;
  std::vector<std::pair<std::string, size_t>> to_save;  // (path, arg index)
  for (size_t i = 0; i < exe.args.size(); ++i) {
    const sava::ArgInfo& arg = exe.args[i];
    uint64_t addr = next_addr;
    next_addr = (next_addr + arg.bytes() + 0xfff) & ~uint64_t(0xfff);
    args.push_back(addr);
    if (arg.kind != sava::ArgKind::Out) {
      if (next_in >= inputs.size()) die("missing -i for argument '" + arg.name + "'");
      sava::npy::Array in;
      if (!sava::npy::load(inputs[next_in], &in, &error)) die(error);
      if (in.descr != npyDescr(arg.dtype))
        die(inputs[next_in] + ": dtype " + in.descr + " does not match argument '" + arg.name +
            "' (" + sava::dtypeName(arg.dtype) + ", expected " + npyDescr(arg.dtype) + ")");
      if (in.data.size() != arg.bytes())
        die(inputs[next_in] + ": size does not match argument '" + arg.name + "'");
      check(sava_dram_write(dev, addr, in.data.data(), in.data.size()));
      ++next_in;
    }
    if (arg.kind != sava::ArgKind::In) {
      if (next_out >= outputs.size()) die("missing -o for argument '" + arg.name + "'");
      to_save.push_back({outputs[next_out++], i});
    }
  }
  if (next_in != inputs.size() || next_out != outputs.size())
    die("more -i/-o files than kernel arguments");

  check(sava_run(dev, args.data(), uint32_t(args.size())));

  for (const auto& [path, i] : to_save) {
    const sava::ArgInfo& arg = exe.args[i];
    sava::npy::Array out;
    out.descr = npyDescr(arg.dtype);
    out.shape.assign(arg.dims.begin(), arg.dims.end());
    out.data.resize(arg.bytes());
    check(sava_dram_read(dev, args[i], out.data.data(), out.data.size()));
    if (!sava::npy::save(path, out, &error)) die(error);
  }

  if (stats) {
    sava_stats s;
    sava_get_stats(dev, &s);
    printStats(s);
  }
  sava_device_destroy(dev);
  return 0;
}
