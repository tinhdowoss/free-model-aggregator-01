// The upstream wires this baseline does not speak: anthropic and gemini, in
// both directions, plus the JSON and SSE response translations.
//
// The trial's providers answer TEXT ONLY (never tool calls) with a plain stop,
// so a response translation carries text, usage and the stop reason. The
// request translation is what the score watches: the provider echoes what it
// parsed (system/user text, tool names in order, max_tokens, stops, image
// bytes, json mode), and the client compares that echo with what it sent — so
// every field is carried through exactly. A JSON-mode request goes to an
// openai or gemini wire only: the anthropic wire cannot carry response_format.
//
// Bodies here are single-turn (one user message, optional system), per the
// workload; the translators still carry multi-turn tool history faithfully.

import { isRecord, type Json, type Wire } from './providers.ts';
import { anthropicToOpenAI, openAIToAnthropic, sseEvent, stopReasonOf } from './wire.ts';

/** A fixed epoch second for translated list/answer ids: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A number, or 0 when the value is not one. */
function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One OpenAI content part as an Anthropic block: text or an inline image. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (m !== null) return { type: 'image', source: { type: 'base64', media_type: m[1] === '' ? 'image/png' : m[1], data: m[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * A Chat Completions request as an Anthropic Messages request naming `model`:
 * the system messages become one system string, image_url parts become image
 * blocks (same bytes), tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, tools/tool_choice/max_tokens/temperature/top_p/stop pass
 * through. response_format cannot be carried: never call this for a JSON-mode
 * request (the planner routes those to an openai or gemini wire).
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (role === 'user') {
      if (typeof raw.content === 'string') {
        if (raw.content !== '') messages.push({ role: 'user', content: raw.content });
      } else if (Array.isArray(raw.content)) {
        const blocks: Json[] = [];
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const block = anthropicBlockOf(part);
          if (block !== null) blocks.push(block);
        }
        if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      }
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        const id = typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown';
        if (typeof call.id === 'string' && call.id !== '') toolNames.set(call.id, name);
        blocks.push({ type: 'tool_use', id, name, input: argumentsOf(call.function.arguments) });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      const content = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: id, content }] });
      continue;
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One message's content as Gemini parts: text, images (same bytes), tool calls. */
function geminiPartsOf(role: string, content: unknown, toolCalls: unknown, toolCallId: string | undefined, toolNames: Map<string, string>): Json[] {
  const parts: Json[] = [];
  if (role === 'tool') {
    const name = toolCallId !== undefined ? (toolNames.get(toolCallId) ?? '') : '';
    const text = typeof content === 'string' ? content : textOf(content);
    parts.push({ functionResponse: { name, response: { content: text } } });
    return parts;
  }
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (m !== null) parts.push({ inlineData: { mimeType: m[1] === '' ? 'image/png' : m[1], data: m[2] } });
        else parts.push({ fileData: { fileUri: url } });
      } else if (part.type === 'image' && isRecord(part.source)) {
        const source = part.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
        } else if (source.type === 'url' && typeof source.url === 'string') parts.push({ fileData: { fileUri: source.url } });
      } else if (part.type === 'tool_use' && typeof part.name === 'string') {
        parts.push({ functionCall: { name: part.name, args: isRecord(part.input) ? part.input : {} } });
      } else if (part.type === 'tool_result') {
        parts.push({ functionResponse: { name: '', response: { content: textOf(part.content) } } });
      }
    }
  }
  const calls = Array.isArray(toolCalls) ? toolCalls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const name = typeof call.function.name === 'string' ? call.function.name : '';
    if (typeof call.id === 'string' && call.id !== '') toolNames.set(call.id, name);
    parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
  }
  return parts;
}

/**
 * Either client wire as a Gemini generateContent body: system becomes
 * systemInstruction, messages become contents (user/model roles, same text and
 * image bytes, tool calls as functionCall, tool results as functionResponse),
 * tools become functionDeclarations, response_format json becomes
 * responseMimeType application/json.
 */
export function toGemini(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  const toolNames = new Map<string, string>();
  const contents: Json[] = [];
  const systemParts: string[] = [];
  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const t = textOf(raw.content);
        if (t !== '') systemParts.push(t);
        continue;
      }
      if (raw.role === 'user') {
        const parts = geminiPartsOf('user', raw.content, undefined, undefined, toolNames);
        if (parts.length > 0) contents.push({ role: 'user', parts });
        continue;
      }
      if (raw.role === 'assistant') {
        const parts = geminiPartsOf('assistant', raw.content, raw.tool_calls, undefined, toolNames);
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role === 'tool') {
        const parts = geminiPartsOf('tool', raw.content, undefined, typeof raw.tool_call_id === 'string' ? raw.tool_call_id : undefined, toolNames);
        if (parts.length > 0) contents.push({ role: 'user', parts });
        continue;
      }
    }
  } else {
    const system = textOf(body.system);
    if (system !== '') systemParts.push(system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const parts = geminiPartsOf(raw.role === 'assistant' ? 'assistant' : 'user', raw.content, undefined, undefined, toolNames);
      if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
  }
  const out: Json = { model, contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  // Tools, in the client's own spelling, as functionDeclarations in order.
  const declarations: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      if (clientApi === 'openai' && isRecord(raw.function) && typeof raw.function.name === 'string') {
        declarations.push({
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        });
      } else if (clientApi === 'anthropic' && typeof raw.name === 'string') {
        declarations.push({
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = clientApi === 'openai' ? (typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined) : typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (clientApi === 'openai') {
    if (typeof body.stop === 'string') config.stopSequences = [body.stop];
    else if (Array.isArray(body.stop)) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
    if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
      config.responseMimeType = 'application/json';
    }
  } else if (Array.isArray(body.stop_sequences)) {
    config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Gemini's finishReason as an OpenAI finish_reason. */
function geminiFinishToOpenAI(finish: unknown, hasFunctionCall: boolean): string {
  if (hasFunctionCall) return 'tool_calls';
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY' || finish === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as an Anthropic stop_reason. */
function geminiFinishToAnthropic(finish: unknown, hasFunctionCall: boolean): string {
  if (hasFunctionCall) return 'tool_use';
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY' || finish === 'RECITATION') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as an OpenAI finish_reason. */
function anthropicFinishToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** The text parts of a Gemini candidate, and whether it called a function. */
function geminiCandidate(candidate: Json): { text: string; calls: Json[]; finish: unknown } {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const calls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') {
      calls.push({
        id: `call_${calls.length}`,
        type: 'function',
        function: { name: part.functionCall.name, arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
      });
    }
  }
  return { text: texts.join(''), calls, finish: candidate.finishReason };
}

/** A Gemini answer as a Chat Completions object. */
export function geminiJsonToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, calls, finish } = geminiCandidate(first);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: geminiFinishToOpenAI(finish, calls.length > 0), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiJsonToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, calls, finish } = geminiCandidate(first);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of calls) {
    if (!isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: geminiFinishToAnthropic(finish, calls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicJsonToOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const calls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    if (block.type === 'tool_use' && typeof block.name === 'string') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: texts.join('') };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicFinishToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One SSE frame's event name and data payload, from a raw frame's lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** The SSE frames of a stream, split on blank lines. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        let event: string | null = null;
        const dataLines: string[] = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
        }
        if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      let event: string | null = null;
      const dataLines: string[] = [];
      for (const line of buffer.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * An OpenAI SSE stream as Anthropic SSE events, buffered so message_start
 * carries the true input_tokens: the usage chunk arrives last on the OpenAI
 * wire, after the finish chunk, so the events are emitted only once the
 * whole upstream stream has been read. (The starter's streaming converter
 * emits message_start on the first chunk, with input_tokens still 0.)
 */
export async function* openAIStreamToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let text = '';
  let finish: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const calls: Json[] = [];
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '[DONE]') break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    const tcalls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of tcalls) {
      if (!isRecord(call)) continue;
      const idx = typeof call.index === 'number' ? call.index : calls.length;
      let target = calls[idx];
      if (target === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        target = {
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${idx}`,
          type: 'function',
          function: { name: typeof fn.name === 'string' ? fn.name : '', arguments: '' },
        };
        calls[idx] = target;
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && isRecord(target.function)) {
        const f = target.function;
        f.arguments = `${typeof f.arguments === 'string' ? f.arguments : ''}${fn.arguments}`;
      }
    }
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  const stopReason = stopReasonOf(finish);
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
  });
  let index = 0;
  yield sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
  if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text } });
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
  index += 1;
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    yield sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: {} },
    });
    const args = typeof call.function.arguments === 'string' ? call.function.arguments : '';
    if (args !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: args } });
    yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}

/** An OpenAI chat chunk. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/** A Gemini stream as OpenAI SSE frames: role chunk, text deltas, finish, usage, [DONE]. */
export async function* geminiStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let text = '';
  let finish: unknown = null;
  let calls: Json[] = [];
  let usage: Json | null = null;
  let started = false;
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '[DONE]') break;
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const parsed = geminiCandidate(first);
    if (parsed.text !== '') {
      if (!started) {
        started = true;
        yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
      }
      yield `data: ${JSON.stringify(openAIChunk(model, { content: parsed.text }, null))}\n\n`;
      text += parsed.text;
    }
    if (parsed.calls.length > 0) calls = parsed.calls;
    if (first.finishReason !== undefined) finish = first.finishReason;
    if (isRecord(value.usageMetadata)) {
      usage = {
        prompt_tokens: numberOrZero(value.usageMetadata.promptTokenCount),
        completion_tokens: numberOrZero(value.usageMetadata.candidatesTokenCount),
        total_tokens: numberOrZero(value.usageMetadata.totalTokenCount),
      };
    }
  }
  if (!started) {
    started = true;
    yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  }
  void text;
  const delta: Json = {};
  if (calls.length > 0) delta.tool_calls = calls;
  yield `data: ${JSON.stringify(openAIChunk(model, delta, geminiFinishToOpenAI(finish, calls.length > 0)))}\n\n`;
  if (usage !== null) {
    yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage })}\n\n`;
  }
  yield 'data: [DONE]\n\n';
}

/** A Gemini stream as Anthropic SSE events. */
export async function* geminiStreamToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let text = '';
  let finish: unknown = null;
  let usage: { input: number; output: number } = { input: 0, output: 0 };
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '[DONE]') break;
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const parsed = geminiCandidate(first);
    if (parsed.text !== '') text += parsed.text;
    if (first.finishReason !== undefined) finish = first.finishReason;
    if (isRecord(value.usageMetadata)) {
      usage = { input: numberOrZero(value.usageMetadata.promptTokenCount), output: numberOrZero(value.usageMetadata.candidatesTokenCount) };
    }
  }
  const stopReason = geminiFinishToAnthropic(finish, false);
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: usage.output } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}

/** An Anthropic stream as OpenAI SSE frames. */
export async function* anthropicStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let text = '';
  let stopReason: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const calls: Json[] = [];
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '[DONE]') break;
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (value.type === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) {
      inputTokens = numberOrZero(value.message.usage.input_tokens);
    }
    if (value.type === 'content_block_delta' && isRecord(value.delta)) {
      if (value.delta.type === 'text_delta' && typeof value.delta.text === 'string') text += value.delta.text;
      if (value.delta.type === 'input_json_delta' && typeof value.delta.partial_json === 'string') {
        // Tool arguments arrive as partial JSON: accumulate on the last call.
        const last = calls[calls.length - 1];
        if (last !== undefined && isRecord(last.function)) {
          last.function.arguments = `${typeof last.function.arguments === 'string' ? last.function.arguments : ''}${value.delta.partial_json}`;
        }
      }
    }
    if (value.type === 'content_block_start' && isRecord(value.content_block) && value.content_block.type === 'tool_use') {
      calls.push({
        id: typeof value.content_block.id === 'string' ? value.content_block.id : 'toolu_unknown',
        type: 'function',
        function: { name: typeof value.content_block.name === 'string' ? value.content_block.name : '', arguments: '' },
      });
    }
    if (value.type === 'message_delta' && isRecord(value.delta) && typeof value.delta.stop_reason === 'string') {
      stopReason = value.delta.stop_reason;
      if (isRecord(value.usage)) outputTokens = numberOrZero(value.usage.output_tokens);
    }
  }
  yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  if (text !== '') yield `data: ${JSON.stringify(openAIChunk(model, { content: text }, null))}\n\n`;
  const delta: Json = {};
  if (calls.length > 0) delta.tool_calls = calls;
  yield `data: ${JSON.stringify(openAIChunk(model, delta, anthropicFinishToOpenAI(stopReason)))}\n\n`;
  yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/**
 * One client request to a provider on its own wire: the URL, headers and body.
 * Returns the fetch Response; the caller translates it to the client's wire.
 */
export async function callUpstream(
  envUrl: string,
  provider: { base: string; wire: Wire; auth: string; key: string },
  model: string,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<Response> {
  const base = provider.base.trim();
  const prefix = envUrl.replace(/\/+$/, '');
  const joined = /^https?:\/\//i.test(base) ? base.replace(/\/+$/, '') : `${prefix}${base.startsWith('/') ? '' : '/'}${base.replace(/\/+$/, '')}`;
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (provider.key !== '') {
    if (provider.auth === 'bearer') headers.authorization = `Bearer ${provider.key}`;
    else if (provider.auth === 'x-api-key') headers['x-api-key'] = provider.key;
    else if (provider.auth === 'x-goog-api-key') headers['x-goog-api-key'] = provider.key;
  }
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstreamBody = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) {
      // The client's stream check wants a final usage chunk; the provider
      // only sends one when include_usage is asked for. The echo does not
      // carry stream_options, so asking changes nothing the client checks.
      upstreamBody.stream_options = { include_usage: true };
    }
    return fetch(`${joined}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstreamBody = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
    return fetch(`${joined}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }
  // gemini: the model and the stream live in the path.
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const upstreamBody = toGemini(clientApi, body, model);
  return fetch(`${joined}/v1beta/models/${encodeURIComponent(model)}:${action}`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

/** A provider's JSON answer translated to the client's wire. */
export function translateJsonAnswer(providerWire: Wire, clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  if (providerWire === 'openai' && clientApi === 'anthropic') return openAIToAnthropic(body, model);
  if (providerWire === 'anthropic' && clientApi === 'openai') return anthropicJsonToOpenAI(body, model);
  if (providerWire === 'gemini' && clientApi === 'openai') return geminiJsonToOpenAI(body, model);
  if (providerWire === 'gemini' && clientApi === 'anthropic') return geminiJsonToAnthropic(body, model);
  return body;
}

/** A provider's SSE stream translated to the client's wire, as SSE frames. */
export function translateStreamAnswer(
  providerWire: Wire,
  clientApi: 'openai' | 'anthropic',
  stream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  if (providerWire === 'gemini' && clientApi === 'openai') return geminiStreamToOpenAI(stream, model);
  if (providerWire === 'gemini' && clientApi === 'anthropic') return geminiStreamToAnthropic(stream, model);
  if (providerWire === 'anthropic' && clientApi === 'openai') return anthropicStreamToOpenAI(stream, model);
  if (providerWire === 'openai' && clientApi === 'anthropic') return openAIStreamToAnthropic(stream, model);
  // Same wire both sides (openai->openai, anthropic->anthropic): pass frames through.
  return (async function* (): AsyncGenerator<string> {
    for await (const frame of sseFrames(stream)) {
      yield `${frame.event === null ? '' : `event: ${frame.event}\n`}data: ${frame.data}\n\n`;
    }
  })();
}
