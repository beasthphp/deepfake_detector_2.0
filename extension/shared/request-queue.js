import { DEFAULT_OPTIONS } from "./constants.js";

export class BoundedRequestQueue {
  constructor({ concurrency = DEFAULT_OPTIONS.scanRequestConcurrency, maxQueued = DEFAULT_OPTIONS.scanQueueLimit, worker, onError } = {}) {
    this.concurrency = clampInteger(concurrency, 1, 6);
    this.maxQueued = clampInteger(maxQueued, 1, 500);
    this.worker = typeof worker === "function" ? worker : async () => {};
    this.onError = typeof onError === "function" ? onError : () => {};
    this.queue = [];
    this.active = new Map();
    this.canceledIds = new Set();
    this.sequence = 0;
  }

  configure({ concurrency, maxQueued } = {}) {
    if (concurrency != null) {
      this.concurrency = clampInteger(concurrency, 1, 6);
    }
    if (maxQueued != null) {
      this.maxQueued = clampInteger(maxQueued, 1, 500);
    }
    this.pump();
  }

  enqueue(item) {
    if (this.queue.length >= this.maxQueued) {
      return { accepted: false, reason: "queue_full", id: "" };
    }
    const id = item?.id || `queue-item-${Date.now()}-${++this.sequence}`;
    const queuedItem = {
      ...(item || {}),
      id,
      priority: Number.isFinite(Number(item?.priority)) ? Number(item.priority) : 0
    };
    this.queue.push(queuedItem);
    this.queue.sort((a, b) => b.priority - a.priority);
    this.pump();
    return { accepted: true, reason: "queued", id };
  }

  cancelWhere(predicate) {
    const shouldCancel = typeof predicate === "function" ? predicate : () => true;
    const kept = [];
    let canceled = 0;
    for (const item of this.queue) {
      if (shouldCancel(item)) {
        this.canceledIds.add(item.id);
        canceled += 1;
      } else {
        kept.push(item);
      }
    }
    this.queue = kept;
    for (const item of this.active.values()) {
      if (shouldCancel(item)) {
        this.canceledIds.add(item.id);
        canceled += 1;
      }
    }
    return canceled;
  }

  cancelAll() {
    return this.cancelWhere(() => true);
  }

  isCanceled(id) {
    return this.canceledIds.has(id);
  }

  stats() {
    return {
      queued: this.queue.length,
      active: this.active.size,
      concurrency: this.concurrency,
      maxQueued: this.maxQueued
    };
  }

  pump() {
    while (this.active.size < this.concurrency && this.queue.length > 0) {
      const item = this.queue.shift();
      if (this.canceledIds.has(item.id)) {
        this.canceledIds.delete(item.id);
        continue;
      }
      this.active.set(item.id, item);
      Promise.resolve()
        .then(() => this.worker(item, { isCanceled: () => this.isCanceled(item.id) }))
        .catch((error) => this.onError(error, item))
        .finally(() => {
          this.active.delete(item.id);
          this.canceledIds.delete(item.id);
          this.pump();
        });
    }
  }
}

function clampInteger(value, min, max) {
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) {
    return min;
  }
  return Math.min(max, Math.max(min, number));
}
