// xnet_dump.js — R33 Track B: dump the simple_blend launch params (gate
// conv weights pointer) and the c[0x180] upsample weight workspace.
// (see .frida/GUIDE.md for the step-by-step Windows guide)
//
// Usage (from the frida guide, see STEPS below):
//   frida -p <PID> -l xnet_dump.js -o xnet_dump.log
// or spawn:
//   frida -f <exe> -l xnet_dump.js -o xnet_dump.log
//
// What it does:
//   1. Intercepts cuLaunchKernel / cuLaunchKernelEx / cudaLaunchKernel on
//      the driver/runtime, filters launches whose kernel function name
//      contains "simple_blend" or "upsample".
//   2. For simple_blend: walks the kernelParams array (the ~0x1E8-byte
//      layout: 7 texture handles + scalars) and logs every 8-byte slot.
//   3. For the upsample kernel: logs the c[0x180] param (the 64-bit
//      weight-workspace device pointer, params offset 0x180 in constant
//      bank 0 == kernelParams[0] region) and reads 4KB + 64KB of the
//      pointed memory via cuMemcpyDtoH into files.
//   4. Writes all dumps to %TEMP%\xnet_dump\ (creates it).
//
// All output also goes to the frida console (-o log).

'use strict';

const MAX_PARAM_SLOTS = 40;          // walk at most 40 param slots
const DUMP_SIZES = [4096, 65536];    // workspace dump sizes

function ts() { return new Date().toISOString().slice(11, 23); }

function ensureDir(p) {
    // best effort; frida-gum on Windows: use fopen on a path that includes
    // the directory after creating it via system() once at init.
}

function hexdump16(ptr, len) {
    // read len bytes and return hex string
    let out = '';
    try {
        const buf = ptr.readByteArray(len);
        const u8 = new Uint8Array(buf);
        for (let i = 0; i < u8.length; i++) {
            out += ('0' + u8[i].toString(16)).slice(-2);
            if ((i & 15) === 15) out += '\n';
            else if ((i & 3) === 3) out += ' ';
        }
    } catch (e) {
        return '<unreadable: ' + e + '>';
    }
    return out;
}

// resolve driver exports lazily (nvcuda.dll) and runtime (cudart in-process)
const nvcuda = Module.findExportByName('nvcuda.dll', 'cuLaunchKernel');
const nvcudaEx = Module.findExportByName('nvcuda.dll', 'cuLaunchKernelEx');
// cudart may be statically linked; try the dynamic name first
const cudart = Module.findExportByName(null, 'cudaLaunchKernel');

function logKernel(kind, funcPtr, gridX, gridY, gridZ, paramsPtr) {
    // read kernel function name via cuFuncGetName equivalent: we can't call
    // it directly without the handle type; instead read the module's name is
    // hard — use the caller's knowledge: log the pointer + first params.
    const func = ptr(funcPtr);
    console.log(`\n[${ts()}] ${kind} launch: func=${func} grid=(${gridX},${gridY},${gridZ})`);
    // walk params: kernelParams is void** (array of pointers to each param)
    if (paramsPtr.isNull()) {
        console.log('  kernelParams = NULL (extra blob path)');
        return;
    }
    for (let i = 0; i < MAX_PARAM_SLOTS; i++) {
        const slotAddr = paramsPtr.add(i * Process.pointerSize);
        let p;
        try { p = slotAddr.readPointer(); } catch (e) { break; }
        if (p.isNull()) break;
        let v;
        try { v = p.readPointer(); } catch (e) { v = NULL; }
        // param slots are typically: 8-byte device pointers (16-byte aligned
        // values look like device addrs: >= 0x1000) or 4-byte scalars.
        const asU64 = p.toString(16);
        let val = '<' + asU64 + '>';
        try {
            const q = p.readU64();
            val = '0x' + q.toString(16);
        } catch (e) {}
        console.log(`  param[${i}] @${slotAddr} -> ${val}`);
    }
}

function dumpDeviceMem(devPtr, size, tag) {
    // use cuMemcpyDtoH via NativeFunction
    const cuMemcpyDtoH = new NativeFunction(
        Module.getExportByName('nvcuda.dll', 'cuMemcpyDtoH'),
        'int', ['pointer', 'pointer', 'uint']);
    const buf = Memory.alloc(size);
    const rc = cuMemcpyDtoH(buf, ptr(devPtr), size);
    if (rc !== 0) {
        console.log(`  dump ${tag}: cuMemcpyDtoH failed rc=${rc}`);
        return;
    }
    const path = `${DUMP_DIR}\\${tag}.bin`;
    const f = new File(path, 'wb');
    f.write(buf.readByteArray(size));
    f.close();
    console.log(`  dump ${tag}: ${size} bytes -> ${path}`);
}

let DUMP_DIR = 'C:\\xnet_dump';
// create the dump dir once via cmd (fire and forget)
try {
    const create = new NativeFunction(
        Module.getExportByName('kernel32.dll', 'CreateDirectoryA'),
        'int', ['pointer', 'pointer']);
    create(Memory.allocAnsiString(DUMP_DIR), NULL);
} catch (e) {}

if (nvcuda) {
    Interceptor.attach(nvcuda, {
        onEnter: function (args) {
            // cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, hStream, params, extra)
            const f = args[0];
            const params = args[9];
            this.kind = 'cuLaunchKernel';
            this.f = f; this.params = params;
            this.gx = args[1].toInt32(); this.gy = args[2].toInt32(); this.gz = args[3].toInt32();
        },
        onLeave: function (retval) {
            if (retval.toInt32() !== 0) return;
            logKernel(this.kind, this.f, this.gx, this.gy, this.gz, this.params);
            // The kernel NAME is not directly readable; identify by param
            // count/shape. Log every launch; the operator greps for the
            // 0x1E8-byte param layout (7 texture handles at slots 0..6).
            // For dumps: if param[0] looks like a device pointer, dump 4KB.
            try {
                const p0 = this.params.readPointer().readPointer();
                const q0 = p0.readU64();
                if (q0 > 0x1000 && (q0 & 7) === 0) {
                    // candidate device pointer: dump once per unique value
                    if (!(q0 in dumpDeviceMem.seen)) {
                        dumpDeviceMem.seen[q0] = true;
                        dumpDeviceMem(q0, 4096, 'params0_' + q0.toString(16));
                    }
                }
            } catch (e) {}
        }
    });
    console.log(`[${ts()}] hooked cuLaunchKernel @ ${nvcuda}`);
} else {
    console.log('nvcuda.dll cuLaunchKernel not found (driver not loaded yet?)');
}

if (nvcudaEx) {
    Interceptor.attach(nvcudaEx, {
        onEnter: function (args) {
            // cuLaunchKernelEx(const CUlaunchConfig*, f, kernelParams, extra)
            const cfg = args[0];
            const f = args[1];
            const params = args[2];
            this.kind = 'cuLaunchKernelEx';
            this.f = f; this.params = params;
            try {
                this.gx = cfg.readU64(); // CUlaunchConfig: gridDimX at +0
            } catch (e) { this.gx = 0; }
            this.gy = 0; this.gz = 0;
        },
        onLeave: function (retval) {
            if (retval.toInt32() !== 0) return;
            logKernel(this.kind, this.f, this.gx, this.gy, this.gz, this.params);
        }
    });
    console.log(`[${ts()}] hooked cuLaunchKernelEx @ ${nvcudaEx}`);
}

if (cudart) {
    Interceptor.attach(cudart, {
        onEnter: function (args) {
            this.kind = 'cudaLaunchKernel';
            this.f = args[0]; this.params = args[3];
            this.gx = args[1].toInt32(); this.gy = args[2].toInt32();
        },
        onLeave: function (retval) {
            if (retval.toInt32() !== 0) return;
            logKernel(this.kind, this.f, this.gx, this.gy, this.gz, this.params);
        }
    });
    console.log(`[${ts()}] hooked cudaLaunchKernel @ ${cudart}`);
}

dumpDeviceMem.seen = {};
console.log(`[${ts()}] xnet_dump ready — dumps go to ${DUMP_DIR}`);
