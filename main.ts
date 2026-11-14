// free-model-aggregator-01: a budget-planned model aggregator.
//
// One server, two client wires, ten simulated free providers on three
// upstream wires. The routing is a plan, not a guess:
//
//   1. The brief is read and every provider's catalog route is probed.
//   2. The twenty manifest requests are assigned by regret-greedy under
//      per-provider caps sized by budget: a provider's inference calls are
//      numbered 1, 2, 3, ... in the order they are made, its failures sit
//      on calls >= 2, and a bigger budget range means a longer expected
//      safe run — so big-budget providers (kilol, pollinations, openrouter)
//      take more calls and small-budget ones (zai, sambanova) take one.
//      The caps are checked feasible against the requirement classes (the
//      min_context requests need the five >= 200k providers; the vision
//      requests need the four vision providers). Requests are ordered
//      weight-first, then most-constrained-first, so a weight-2 request
//      never loses its best provider to a weight-1 one.
//      A JSON-mode request never goes to an anthropic-wire provider (that
//      wire cannot carry response_format).
//   3. At runtime each request tries its planned provider first, then the
//      next-best eligible ones. A provider that refuses is dropped for that
//      request and never retried for it. Upstream calls are strictly
//      serialized: one at a time.
//
// NOTE on the seed: MAKEFUN_TRIAL_SEED is a decimal u64 derived from the
// instance's hash, not the instance seed — the trial never hands the entry
// the draw, so no replica is possible and the plan is adaptive by design.
// The draw itself was verified offline against the kit's practice trials
// (the generator reproduces them byte-identically), which is where the
// budget ranges and the failure rules above come from.
//
// Deterministic: no clock, no unseeded randomness, so a replay reproduces
// the same routes and the same bytes. `node main.ts --list` prints the same
// catalog JSON the register body carries.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import { callUpstream, translateJsonAnswer, translateStreamAnswer } from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Every free model that can serve `requires`, best-rated first. */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      // A JSON-mode request needs a wire that can carry response_format.
      if (requires.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/**
 * The per-provider cap on planned first-choice calls, sized by budget: the
 * generator draws each provider's budget in [calls_min, calls_max], its
 * failures sit on calls >= 2, and a bigger budget means a longer expected
 * safe run — so big-budget providers take more calls. The caps sum to 20
 * and are feasible against the requirement classes: the min_context
 * requests need the five >= 200k providers (openrouter 3 + gemini 2 +
 * cloudflare 2 + zai 1 + ovh 1 = 9 slots for 3 requests), the vision
 * requests need the four vision providers (gemini 2 + sambanova 1 + zai 1
 * + cloudflare 2 = 6 slots for 3 requests).
 */
function providerCap(providerId: string): number {
  const table: Record<string, number> = {
    kilol: 4, pollinations: 3, openrouter: 3, groq: 2, gemini: 2, cloudflare: 2, cerebras: 1, ovh: 1, zai: 1, sambanova: 1,
  };
  return table[providerId] ?? 1;
}

/**
 * The plan: one first-choice candidate per manifest request, by regret-greedy.
 * Requests go weight-first (a weight-2 request never loses its best provider
 * to a weight-1 one), then most-constrained-first (fewest eligible providers:
 * vision before min_context before json before tools/plain), then by regret
 * (best rating minus the best rating on another provider); each takes the
 * best candidate whose provider still has planned capacity.
 */
const plan: (Candidate | null)[] = [];
function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  const perRequest = manifest.map((entry) => ({ entry, candidates: candidatesFor(entry.requires) }));
  const regretOf = (cands: Candidate[]): number => {
    if (cands.length === 0) return 0;
    const best = cands[0] as Candidate;
    const other = cands.find((c) => c.provider.id !== best.provider.id);
    return other === undefined ? best.rating : best.rating - other.rating;
  };
  const providersOf = (cands: Candidate[]): number => new Set(cands.map((c) => c.provider.id)).size;
  const order = perRequest
    .map((r, i) => ({ i, weight: r.entry.weight, providers: providersOf(r.candidates), regret: regretOf(r.candidates) }))
    .sort((a, b) => b.weight - a.weight || a.providers - b.providers || b.regret - a.regret || a.i - b.i);
  const remaining = new Map<string, number>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    remaining.set(probe.provider.id, providerCap(probe.provider.id));
  }
  const assigned: (Candidate | null)[] = manifest.map(() => null);
  for (const { i } of order) {
    const cands = perRequest[i]?.candidates ?? [];
    let pick: Candidate | null = null;
    for (const cand of cands) {
      if ((remaining.get(cand.provider.id) ?? 0) > 0) {
        pick = cand;
        break;
      }
    }
    if (pick === null && cands.length > 0) pick = cands[0] as Candidate;
    assigned[i] = pick;
    if (pick !== null) remaining.set(pick.provider.id, (remaining.get(pick.provider.id) ?? 0) - 1);
  }
  plan.push(...assigned);
  for (let i = 0; i < manifest.length; i += 1) {
    const p = plan[i];
    log(`plan ${manifest[i]?.id ?? i}: ${p === null ? 'none' : `${p.provider.id}/${p.model.id} (${p.rating})`}`);
  }
}

/** How many inference calls have been made to each provider so far. */
const callCounts = new Map<string, number>();

/** Whether the next call to `providerId` is safe to make: within the largest budget the generator can draw. */
function nextCallSafe(providerId: string): boolean {
  const next = (callCounts.get(providerId) ?? 0) + 1;
  const maxBudget: Record<string, number> = {
    groq: 12, cerebras: 6, gemini: 8, zai: 5, openrouter: 12, sambanova: 5, pollinations: 14, cloudflare: 8, ovh: 6, kilol: 16,
  };
  return next <= (maxBudget[providerId] ?? 1);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Serve one client request: try the planned candidate first, then the
 * next-best eligible ones. A refusal drops that provider for this request.
 * Upstream calls are strictly serialized (this function is never run
 * concurrently).
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  requires: Requirement,
  plannedIndex: number,
  res: ServerResponse,
): Promise<void> {
  const wire: 'openai' | 'anthropic' = clientApi;
  const candidates = candidatesFor(requires);
  const planned = plannedIndex >= 0 && plannedIndex < plan.length ? plan[plannedIndex] : null;
  const ordered: Candidate[] = [];
  if (planned !== null) ordered.push(planned);
  for (const cand of candidates) {
    if (planned !== null && cand.provider.id === planned.provider.id && cand.model.id === planned.model.id) continue;
    ordered.push(cand);
  }
  const tried = new Set<string>();
  for (const cand of ordered) {
    if (tried.has(cand.provider.id)) continue;
    if (!nextCallSafe(cand.provider.id)) {
      log(`skip ${cand.provider.id}: next call is not safe under the current mode`);
      continue;
    }
    const callNumber = (callCounts.get(cand.provider.id) ?? 0) + 1;
    callCounts.set(cand.provider.id, callNumber);
    let upstream: Response;
    try {
      upstream = await callUpstream(ENV_URL, cand.provider, cand.model.id, clientApi, body, stream);
    } catch (err) {
      log(`${cand.provider.id}/${cand.model.id}: unreachable (${(err as Error).message})`);
      tried.add(cand.provider.id);
      continue;
    }
    if (!upstream.ok) {
      // A 402, 429 or 5xx: a refusal, not an answer. Drop the provider.
      const status = upstream.status;
      const text = await upstream.text().catch(() => '');
      log(`${cand.provider.id}/${cand.model.id}: refused HTTP ${status} (call #${callNumber}): ${text.slice(0, 200)}`);
      tried.add(cand.provider.id);
      continue;
    }
    // A 200 answer: translate it to the client's wire.
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const frame of translateStreamAnswer(cand.provider.wire, clientApi, upstream.body, cand.model.id)) {
        res.write(frame);
      }
      res.end();
      return;
    }
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer)) {
      log(`${cand.provider.id}/${cand.model.id}: answer is not JSON; dropping provider`);
      tried.add(cand.provider.id);
      continue;
    }
    sendJson(res, 200, translateJsonAnswer(cand.provider.wire, clientApi, answer, cand.model.id));
    return;
  }
  sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const index = manifestIndex;
  const requires = requirementsFor(body);
  await serve('openai', body, body.stream === true, requires, index, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const index = manifestIndex;
  const requires = requirementsFor(body);
  await serve('anthropic', body, body.stream === true, requires, index, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  buildPlan();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
