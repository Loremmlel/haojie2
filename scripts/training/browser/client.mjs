import * as ort from '/ort.mjs';

const status = document.querySelector('#status');
const buttons = [...document.querySelectorAll('button')];
const reportElement = document.querySelector('#report');
const rows = document.querySelector('#rows');
const report = { format: 'haojie-browser-result-v1', started: new Date().toISOString(), runs: [] };
let cases, manifest;
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const save = () => {
  reportElement.textContent = JSON.stringify(report, null, 2);
};
const binary = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
};
const sha256 = async (bytes) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');

function inputs(example) {
  return Object.fromEntries(
    Object.entries(example.inputs).map(([key, tensor]) => {
      const data =
        tensor.type === 'int64'
          ? BigInt64Array.from(tensor.data, BigInt)
          : tensor.type === 'bool'
            ? Uint8Array.from(tensor.data)
            : Float32Array.from(tensor.data);
      return [key, new ort.Tensor(tensor.type, data, tensor.dims)];
    }),
  );
}

function parity(output, example) {
  const width = example.inputs.candidates.dims[1];
  const mask = example.inputs.candidate_mask.data;
  let maxLogit = 0,
    maxProbability = 0,
    top1Equal = 0;
  const probabilities = (values) => {
    const highest = Math.max(...values);
    const weights = values.map((value) => Math.exp(value - highest));
    const total = weights.reduce((a, b) => a + b, 0);
    return weights.map((weight) => weight / total);
  };
  for (let row = 0; row < output.value.length; row++) {
    const logits = output.logits.slice(row * width, (row + 1) * width);
    const expected = example.reference.logits.slice(row * width, (row + 1) * width);
    const actualProbability = probabilities(logits),
      expectedProbability = probabilities(expected);
    if (logits.indexOf(Math.max(...logits)) === expected.indexOf(Math.max(...expected)))
      top1Equal++;
    for (let col = 0; col < width; col++) {
      if (mask[row * width + col])
        maxLogit = Math.max(maxLogit, Math.abs(logits[col] - expected[col]));
      maxProbability = Math.max(
        maxProbability,
        Math.abs(actualProbability[col] - expectedProbability[col]),
      );
    }
  }
  return {
    finite: [...output.logits, ...output.value].every(Number.isFinite),
    max_logit_abs: maxLogit,
    max_probability_abs: maxProbability,
    max_value_abs: Math.max(
      ...output.value.map((v, i) => Math.abs(v - example.reference.value[i])),
    ),
    top1_equal: top1Equal,
    samples: output.value.length,
  };
}

async function forward(session, feeds) {
  const start = performance.now();
  const output = await session.run(feeds);
  try {
    // 显式回读到CPU，防止只计GPU提交时间；转换与释放位于计时边界之后。
    const logits = await output.logits.getData(),
      value = await output.value.getData();
    const elapsed = performance.now() - start;
    return { elapsed, output: { logits: Array.from(logits), value: Array.from(value) } };
  } finally {
    Object.values(output).forEach((tensor) => tensor.dispose());
  }
}

async function run(backend, precision, audit = false) {
  buttons.forEach((button) => {
    button.disabled = true;
  });
  const record = { backend, precision, audit, cases: [] };
  report.runs.push(record);
  let session;
  // 原生WebGPU后端不提供JSEP的JS计时回调；只在非计时审计中收集节点分配日志。
  const originalError = console.error;
  if (audit) {
    record.node_assignments = [];
    console.error = (...args) => {
      const line = args.join(' ');
      if (line.includes('Node(s) placed on') || line.includes('All nodes placed on'))
        record.node_assignments.push(line);
      originalError.apply(console, args);
    };
  }
  try {
    if (backend === 'webgpu' && !report.environment.adapter)
      throw new Error('没有可用WebGPU适配器');
    if (precision === 'mixed-fp16' && !report.environment.shader_f16)
      throw new Error('设备不支持shader-f16');
    status.textContent = `${backend} ${precision}：加载模型…`;
    await tick();
    const start = performance.now(),
      bytes = await binary(`/${precision}.onnx`);
    record.load_ms = performance.now() - start;
    record.model_sha256 = await sha256(bytes);
    if (record.model_sha256 !== manifest.models[precision].sha256)
      throw new Error('模型指纹不匹配');
    const kernels = {};
    ort.env.webgpu.profiling = {
      mode: audit ? 'default' : 'off',
      ondata: (data) => {
        kernels[data.kernelType] = (kernels[data.kernelType] ?? 0) + 1;
      },
    };
    const compileStart = performance.now();
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: [backend],
      ...(audit ? { logSeverityLevel: 0, logVerbosityLevel: 1 } : {}),
    });
    record.session_create_ms = performance.now() - compileStart;
    if (backend === 'webgpu') {
      const device = await ort.env.webgpu.device;
      record.device = {
        vendor: device.adapterInfo?.vendor,
        architecture: device.adapterInfo?.architecture,
        description: device.adapterInfo?.description,
        features: [...device.features],
      };
    }
    for (const example of audit ? cases.slice(1, 2) : cases) {
      status.textContent = `${backend} ${precision}：${example.name} ${audit ? '算子审计' : '计时'}…`;
      await tick();
      const feeds = inputs(example);
      try {
        const first = await forward(session, feeds);
        const result = {
          name: example.name,
          shape: [
            example.inputs.entities.dims[0],
            example.inputs.entities.dims[1],
            example.inputs.candidates.dims[1],
          ],
          first_ms: first.elapsed,
          parity: parity(first.output, example),
          times_ms: [],
        };
        record.cases.push(result);
        if (!result.parity.finite) throw new Error('输出非有限数值');
        if (!audit) {
          for (let i = 0; i < 3; i++) await forward(session, feeds);
          for (let i = 0; i < 20; i++)
            result.times_ms.push((await forward(session, feeds)).elapsed);
          const sorted = [...result.times_ms].sort((a, b) => a - b);
          result.median_ms = (sorted[9] + sorted[10]) / 2;
          result.p95_ms = sorted[18];
          const row = rows.insertRow();
          for (const cell of [
            backend + ' ' + precision,
            result.shape.join('×'),
            result.first_ms.toFixed(1),
            result.median_ms.toFixed(1),
            result.p95_ms.toFixed(1),
            result.parity.max_probability_abs.toExponential(2),
            `${result.parity.top1_equal}/${result.parity.samples}`,
          ])
            row.insertCell().textContent = cell;
        }
        save();
      } finally {
        Object.values(feeds).forEach((tensor) => tensor.dispose());
      }
    }
    if (audit) {
      await (await ort.env.webgpu.device).queue.onSubmittedWorkDone();
      await tick();
      record.gpu_kernels = kernels;
      record.profiling_note = Object.keys(kernels).length
        ? '收到GPU算子计时回调'
        : '此运行库没有返回JS算子计时；查看控制台详细节点分配，不把空结果当作GPU证明';
    }
    record.complete = true;
    status.textContent = `${backend} ${precision}完成${audit ? '；' + record.profiling_note : '，结果见下表'}。`;
    if (record.node_assignments?.length)
      status.textContent += '\n' + record.node_assignments.join('\n');
  } catch (error) {
    record.error = String(error.stack ?? error);
    status.textContent = `实验失败：${error.message ?? error}`;
  } finally {
    console.error = originalError;
    await session?.release();
    save();
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

try {
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'warning';
  const adapter = await navigator.gpu?.requestAdapter();
  report.environment = {
    url: location.href,
    user_agent: navigator.userAgent,
    secure_context: isSecureContext,
    cross_origin_isolated: crossOriginIsolated,
    hardware_concurrency: navigator.hardwareConcurrency,
    ort: ort.env.versions,
    wasm_threads: 1,
    adapter: adapter
      ? {
          vendor: adapter.info?.vendor,
          architecture: adapter.info?.architecture,
          description: adapter.info?.description,
          is_fallback: adapter.info?.isFallbackAdapter,
        }
      : null,
    shader_f16: adapter?.features.has('shader-f16') ?? false,
  };
  document.querySelector('#capabilities').textContent = JSON.stringify(report.environment, null, 2);
  const caseBytes = await binary('/cases.json');
  cases = JSON.parse(new TextDecoder().decode(caseBytes));
  manifest = await (await fetch('/manifest.json')).json();
  if ((await sha256(caseBytes)) !== manifest.cases_sha256) throw new Error('真实输入指纹不匹配');
  report.manifest = manifest;
  const start = performance.now();
  ort.env.wasm.wasmBinary = await binary('/runtime.wasm');
  report.runtime_load_ms = performance.now() - start;
  report.runtime_bytes = ort.env.wasm.wasmBinary.byteLength;
  report.method = {
    warmup: 3,
    measured: 20,
    cpu_inputs: true,
    cpu_output_readback: true,
    includes_encoding: false,
    includes_action_decoding: false,
    includes_mcts: false,
  };
  document.querySelector('#cpu').onclick = () => run('wasm', 'fp32');
  document.querySelector('#gpu32').onclick = () => run('webgpu', 'fp32');
  document.querySelector('#gpu16').onclick = () => run('webgpu', 'mixed-fp16');
  document.querySelector('#audit').onclick = () => run('webgpu', 'mixed-fp16', true);
  buttons.forEach((button) => {
    button.disabled = false;
  });
  status.textContent = '准备完成。请逐组运行，避免其他任务占用CPU/GPU。';
  save();
} catch (error) {
  report.error = String(error.stack ?? error);
  status.textContent = `准备失败：${error.message}`;
  save();
}
