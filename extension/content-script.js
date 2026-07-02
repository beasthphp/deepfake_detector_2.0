(() => {
  if (window.__DFD_STATUS_SCRIPT_LOADED__) {
    return;
  }
  window.__DFD_STATUS_SCRIPT_LOADED__ = true;

  const TOAST_ID = "dfd-manual-analysis-toast";
  const OVERLAY_ROOT_ID = "dfd-page-scan-overlay-root";
  const STYLE_ID = "dfd-page-scan-style";
  const REMOVE_DELAY_MS = 6500;
  const DEFAULT_OPTIONS = {
    minDisplayedWidth: 128,
    minDisplayedHeight: 128,
    minNaturalWidth: 128,
    minNaturalHeight: 128,
    scanRootMarginPx: 300,
    overlaysEnabled: true,
    diagnosticsEnabled: false,
    showModelScores: true,
    mockModeEnabled: false,
    apiBaseUrl: "http://127.0.0.1:8000"
  };

  const IMAGE_SCAN_STATE = {
    notScanned: "not_scanned",
    queued: "queued",
    downloading: "downloading",
    analysing: "analysing",
    completed: "completed",
    noFaceDetected: "no_face_detected",
    unsupported: "unsupported",
    failed: "failed",
    apiOffline: "api_offline"
  };

  const state = {
    active: false,
    sessionId: "",
    options: { ...DEFAULT_OPTIONS },
    backendMetadata: null,
    imageSequence: 0,
    imageRecords: new WeakMap(),
    recordsById: new Map(),
    submittedLocalSignatures: new Set(),
    intersectionObserver: null,
    mutationObserver: null,
    resizeObserver: null,
    pendingMutations: [],
    mutationTimer: 0,
    repositionFrame: 0,
    listenersAttached: false,
    diagnostics: createDiagnostics()
  };

  chrome.runtime.onMessage.addListener((message) => {
    if (!message || typeof message.type !== "string") {
      return;
    }
    if (message.type === "DFD_STATUS") {
      showToast(message.message || "Deepfake analysis update", message.state || "running");
    } else if (message.type === "DFD_SCAN_TOAST") {
      showToast(message.message || "Page scan update", message.state || "running");
    } else if (message.type === "DFD_SCAN_START") {
      startScanner(message);
    } else if (message.type === "DFD_SCAN_STOP") {
      stopScanner(message.reason || "stopped");
    } else if (message.type === "DFD_SCAN_CLEAR") {
      clearPageResults();
    } else if (message.type === "DFD_SCAN_IMAGE_STATE") {
      applyImageState(message);
    } else if (message.type === "DFD_SCAN_IMAGE_RESULT") {
      applyImageResult(message);
    } else if (message.type === "DFD_SCAN_IMAGE_ERROR") {
      applyImageError(message);
    }
  });

  function startScanner(message) {
    if (document.querySelector('input[type="password"]')) {
      showToast("Scanning disabled on pages with password fields", "failed");
      return;
    }
    state.active = true;
    state.sessionId = String(message.sessionId || "");
    state.options = { ...DEFAULT_OPTIONS, ...(message.options || {}) };
    state.backendMetadata = message.backendMetadata || null;
    state.submittedLocalSignatures.clear();
    state.diagnostics = createDiagnostics();
    ensureOverlayRoot();
    ensureStyle();
    attachRepositionListeners();
    setupObservers();
    resetRecordsForNewSession();
    discoverImages(document);
    showToast(
      state.options.mockModeEnabled ? "Mock page scanning enabled" : "Visible image scanning enabled",
      "running"
    );
    updateDiagnosticsPanel();
  }

  function stopScanner(reason) {
    state.active = false;
    disconnectDiscoveryObservers();
    showToast(reason === "global_stop" ? "Global scanning stopped" : "Page scanning stopped", "running");
    updateDiagnosticsPanel();
  }

  function clearPageResults() {
    for (const record of state.recordsById.values()) {
      removeRecordOverlays(record);
      record.response = null;
      record.error = null;
      record.state = IMAGE_SCAN_STATE.notScanned;
      record.cacheHit = false;
    }
    const diagnostics = document.getElementById("dfd-diagnostics-panel");
    if (diagnostics) {
      diagnostics.remove();
    }
    showToast("Page scan results cleared", "running");
  }

  function setupObservers() {
    disconnectDiscoveryObservers();
    state.intersectionObserver = new IntersectionObserver(handleIntersections, {
      root: null,
      rootMargin: `${Math.max(0, Number(state.options.scanRootMarginPx) || 0)}px`,
      threshold: 0.01
    });
    state.mutationObserver = new MutationObserver(queueMutations);
    state.mutationObserver.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src", "srcset", "sizes", "style", "class", "hidden", "loading"]
    });
    if (!state.resizeObserver) {
      state.resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.target instanceof HTMLImageElement) {
            const record = state.imageRecords.get(entry.target);
            if (record) {
              evaluateImage(entry.target, "resize");
              renderRecordOverlay(record);
            }
          }
        }
      });
    }
  }

  function disconnectDiscoveryObservers() {
    if (state.intersectionObserver) {
      state.intersectionObserver.disconnect();
      state.intersectionObserver = null;
    }
    if (state.mutationObserver) {
      state.mutationObserver.disconnect();
      state.mutationObserver = null;
    }
    clearTimeout(state.mutationTimer);
    state.pendingMutations = [];
  }

  function discoverImages(root) {
    const images = [];
    if (root instanceof HTMLImageElement) {
      images.push(root);
    }
    if (root?.querySelectorAll) {
      images.push(...root.querySelectorAll("img"));
    }
    for (const img of images) {
      registerImage(img);
    }
    updateDiagnosticsPanel();
  }

  function registerImage(img) {
    if (!(img instanceof HTMLImageElement) || state.imageRecords.has(img)) {
      return state.imageRecords.get(img) || null;
    }
    const record = {
      id: `dfd-img-${++state.imageSequence}`,
      img,
      state: IMAGE_SCAN_STATE.notScanned,
      localSignature: "",
      submittedSignature: "",
      sourceKey: "",
      response: null,
      error: null,
      cacheHit: false,
      isNearViewport: false,
      overlayNodes: []
    };
    state.imageRecords.set(img, record);
    state.recordsById.set(record.id, record);
    state.diagnostics.imagesDiscovered += 1;
    img.addEventListener("load", () => evaluateImage(img, "load"));
    img.addEventListener("error", () => markUnsupported(record, "image_load_error"));
    if (state.intersectionObserver) {
      state.intersectionObserver.observe(img);
    }
    if (state.resizeObserver) {
      state.resizeObserver.observe(img);
    }
    evaluateImage(img, "discovered");
    return record;
  }

  function handleIntersections(entries) {
    for (const entry of entries) {
      const record = state.imageRecords.get(entry.target);
      if (!record) {
        continue;
      }
      record.isNearViewport = entry.isIntersecting || entry.intersectionRatio > 0;
      if (record.isNearViewport) {
        evaluateImage(entry.target, "visible");
      }
    }
  }

  function queueMutations(mutations) {
    state.pendingMutations.push(...mutations);
    if (state.mutationTimer) {
      return;
    }
    state.mutationTimer = setTimeout(processQueuedMutations, 120);
  }

  function processQueuedMutations() {
    const mutations = state.pendingMutations.splice(0);
    state.mutationTimer = 0;
    for (const mutation of mutations) {
      if (mutation.type === "childList") {
        for (const node of mutation.addedNodes) {
          discoverImages(node);
        }
        for (const node of mutation.removedNodes) {
          cleanupRemovedImages(node);
        }
      } else if (mutation.type === "attributes" && mutation.target instanceof HTMLImageElement) {
        resetImageForPotentialSourceChange(mutation.target);
      }
    }
    updateDiagnosticsPanel();
  }

  function cleanupRemovedImages(root) {
    const images = [];
    if (root instanceof HTMLImageElement) {
      images.push(root);
    }
    if (root?.querySelectorAll) {
      images.push(...root.querySelectorAll("img"));
    }
    for (const img of images) {
      const record = state.imageRecords.get(img);
      if (!record) {
        continue;
      }
      removeRecordOverlays(record);
      if (state.intersectionObserver) {
        state.intersectionObserver.unobserve(img);
      }
      if (state.resizeObserver) {
        state.resizeObserver.unobserve(img);
      }
      chrome.runtime.sendMessage({
        type: "DFD_SCAN_IMAGE_REMOVED",
        imageId: record.id,
        sourceKey: record.sourceKey
      });
      state.recordsById.delete(record.id);
      state.imageRecords.delete(img);
    }
  }

  function resetImageForPotentialSourceChange(img) {
    const record = registerImage(img);
    const descriptor = buildDescriptor(img, record.id);
    if (descriptor.localSignature !== record.localSignature) {
      removeRecordOverlays(record);
      record.localSignature = descriptor.localSignature;
      record.submittedSignature = "";
      record.sourceKey = "";
      record.response = null;
      record.error = null;
      record.cacheHit = false;
      record.state = IMAGE_SCAN_STATE.notScanned;
      evaluateImage(img, "source_changed");
    } else {
      renderRecordOverlay(record);
    }
  }

  function evaluateImage(img, reason) {
    if (!state.active) {
      return;
    }
    const record = registerImage(img);
    const descriptor = buildDescriptor(img, record.id);
    record.localSignature = descriptor.localSignature;

    const eligibility = evaluateLocalEligibility(descriptor, state.options);
    if (!eligibility.eligible) {
      if (["unsupported_url", "image_load_error"].includes(eligibility.reason)) {
        markUnsupported(record, eligibility.reason);
      } else {
        state.diagnostics.imagesSkipped += 1;
      }
      return;
    }
    if (!record.isNearViewport && reason !== "visible") {
      return;
    }
    if (record.submittedSignature === descriptor.localSignature) {
      return;
    }

    record.submittedSignature = descriptor.localSignature;
    record.state = IMAGE_SCAN_STATE.queued;
    record.error = null;
    record.response = null;
    state.diagnostics.imagesEligible += 1;
    state.diagnostics.cacheMisses += 1;
    renderRecordOverlay(record);
    chrome.runtime.sendMessage({
      type: "DFD_SCAN_IMAGE_DISCOVERED",
      sessionId: state.sessionId,
      image: descriptor,
      priority: record.isNearViewport ? 100 : 50
    });
  }

  function buildDescriptor(img, id) {
    const rect = img.getBoundingClientRect();
    const currentSrc = img.currentSrc || img.src || "";
    const displayedWidth = Math.max(0, rect.width);
    const displayedHeight = Math.max(0, rect.height);
    const visible = isElementVisible(img, rect);
    const localSignature = [
      currentSrc,
      img.naturalWidth || 0,
      img.naturalHeight || 0,
      Math.round(displayedWidth),
      Math.round(displayedHeight)
    ].join("|");
    return {
      id,
      src: img.src || "",
      currentSrc,
      naturalWidth: img.naturalWidth || 0,
      naturalHeight: img.naturalHeight || 0,
      displayedWidth,
      displayedHeight,
      pageX: rect.left + window.scrollX,
      pageY: rect.top + window.scrollY,
      viewportX: rect.left,
      viewportY: rect.top,
      isLoaded: Boolean(img.complete && img.naturalWidth && img.naturalHeight),
      loadingState: img.complete ? "complete" : "loading",
      visibilityState: visible ? "visible" : "hidden",
      isVisible: visible,
      localSignature,
      devicePixelRatio: window.devicePixelRatio || 1
    };
  }

  function evaluateLocalEligibility(descriptor, options) {
    if (!isHttpImageUrl(descriptor.currentSrc || descriptor.src)) {
      return { eligible: false, reason: "unsupported_url" };
    }
    if (!descriptor.isLoaded) {
      return { eligible: false, reason: "not_loaded" };
    }
    if (!descriptor.isVisible) {
      return { eligible: false, reason: "hidden" };
    }
    if (
      descriptor.displayedWidth <= 1 ||
      descriptor.displayedHeight <= 1 ||
      descriptor.naturalWidth <= 1 ||
      descriptor.naturalHeight <= 1
    ) {
      return { eligible: false, reason: "tracking_pixel" };
    }
    if (descriptor.displayedWidth < options.minDisplayedWidth || descriptor.displayedHeight < options.minDisplayedHeight) {
      return { eligible: false, reason: "display_too_small" };
    }
    if (descriptor.naturalWidth < options.minNaturalWidth || descriptor.naturalHeight < options.minNaturalHeight) {
      return { eligible: false, reason: "natural_too_small" };
    }
    if (hasExtremeAspectRatio(descriptor.displayedWidth, descriptor.displayedHeight)) {
      return { eligible: false, reason: "extreme_aspect_ratio" };
    }
    if (isLikelyIconOrLogo(descriptor.currentSrc || descriptor.src, descriptor.naturalWidth, descriptor.naturalHeight)) {
      return { eligible: false, reason: "likely_icon_or_logo" };
    }
    return { eligible: true, reason: "eligible" };
  }

  function applyImageState(message) {
    const record = recordFromMessage(message);
    if (!record) {
      return;
    }
    record.sourceKey = message.sourceKey || record.sourceKey;
    record.state = message.state || IMAGE_SCAN_STATE.queued;
    record.error = null;
    if (record.state === IMAGE_SCAN_STATE.queued) {
      state.diagnostics.queuedRequests += 1;
    }
    renderRecordOverlay(record);
    updateDiagnosticsPanel();
  }

  function applyImageResult(message) {
    const record = recordFromMessage(message);
    if (!record) {
      return;
    }
    record.sourceKey = message.sourceKey || record.sourceKey;
    record.state = message.state || IMAGE_SCAN_STATE.completed;
    record.response = message.response || null;
    record.error = null;
    record.cacheHit = Boolean(message.cacheHit);
    if (record.cacheHit) {
      state.diagnostics.cacheHits += 1;
    }
    if (record.response?.status === "no_face_detected") {
      state.diagnostics.noFaceResults += 1;
    } else {
      state.diagnostics.completedAnalyses += 1;
    }
    const total = Number(record.response?.timing_ms?.total);
    if (Number.isFinite(total)) {
      state.diagnostics.apiTimeTotal += total;
      state.diagnostics.apiTimeCount += 1;
    }
    renderRecordOverlay(record);
    updateDiagnosticsPanel();
  }

  function applyImageError(message) {
    const record = recordFromMessage(message);
    if (!record) {
      return;
    }
    record.sourceKey = message.sourceKey || record.sourceKey;
    record.state = message.state || IMAGE_SCAN_STATE.failed;
    record.error = message.error || { message: "Image scan failed." };
    record.response = null;
    state.diagnostics.failedAnalyses += 1;
    renderRecordOverlay(record);
    updateDiagnosticsPanel();
  }

  function recordFromMessage(message) {
    if (message.sessionId !== state.sessionId) {
      return null;
    }
    const record = state.recordsById.get(message.imageId);
    if (!record) {
      return null;
    }
    if (message.localSignature && message.localSignature !== record.localSignature) {
      return null;
    }
    if (record.sourceKey && message.sourceKey && record.sourceKey !== message.sourceKey) {
      return null;
    }
    return record;
  }

  function markUnsupported(record, reason) {
    record.state = IMAGE_SCAN_STATE.unsupported;
    record.error = { message: reason };
    state.diagnostics.imagesSkipped += 1;
    removeRecordOverlays(record);
  }

  function renderRecordOverlay(record) {
    removeRecordOverlays(record);
    if (!state.options.overlaysEnabled || !record.img.isConnected) {
      return;
    }
    const rect = record.img.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0 || rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) {
      return;
    }
    if ([IMAGE_SCAN_STATE.queued, IMAGE_SCAN_STATE.downloading, IMAGE_SCAN_STATE.analysing].includes(record.state)) {
      record.overlayNodes.push(createBadge(rect, statusText(record.state), "running"));
      return;
    }
    if (record.state === IMAGE_SCAN_STATE.noFaceDetected || record.response?.status === "no_face_detected") {
      record.overlayNodes.push(createBadge(rect, "No face detected · Experimental detector", "neutral"));
      return;
    }
    if (record.state === IMAGE_SCAN_STATE.failed || record.state === IMAGE_SCAN_STATE.apiOffline) {
      record.overlayNodes.push(createBadge(rect, record.state === IMAGE_SCAN_STATE.apiOffline ? "API offline" : "Scan failed", "failed"));
      return;
    }
    if (record.state !== IMAGE_SCAN_STATE.completed || !record.response) {
      return;
    }

    const faces = Array.isArray(record.response.faces) ? record.response.faces : [];
    record.overlayNodes.push(
      createBadge(rect, `${record.response.mock ? "Mock mode" : "Experimental detector"} · ${faces.length} ${faces.length === 1 ? "face" : "faces"} analysed`, "completed")
    );

    const computed = getComputedStyle(record.img);
    const geometry = computeRenderedImageGeometry({
      naturalWidth: record.img.naturalWidth,
      naturalHeight: record.img.naturalHeight,
      displayedWidth: rect.width,
      displayedHeight: rect.height,
      objectFit: computed.objectFit,
      objectPosition: computed.objectPosition
    });
    for (const face of faces) {
      const mapped = mapFaceBoxToDisplayedImage(face.bounding_box, geometry);
      if (!mapped.visible) {
        continue;
      }
      record.overlayNodes.push(createFaceOverlay(record, face, rect, mapped));
    }
  }

  function createFaceOverlay(record, face, rect, mapped) {
    const root = ensureOverlayRoot();
    const box = document.createElement("div");
    box.className = `dfd-face-box ${labelClass(face.label)}`;
    box.style.left = `${rect.left + mapped.x}px`;
    box.style.top = `${rect.top + mapped.y}px`;
    box.style.width = `${mapped.width}px`;
    box.style.height = `${mapped.height}px`;

    const label = document.createElement("div");
    label.className = "dfd-face-label";
    label.textContent = faceLabelText(face, record.response);
    if (mapped.y < 26) {
      label.classList.add("inside");
    }
    box.append(label);

    const details = [
      `Experimental detector`,
      `Face ${Number(face.face_index) + 1}`,
      `Label ${face.label}`,
      `Real score ${formatScore(face.real_score)}`,
      `Fake score ${formatScore(face.fake_score)}`,
      `Detection score ${face.face_detection_score == null ? "n/a" : formatScore(face.face_detection_score)}`,
      `Model ${record.response?.model?.id || "unknown"} ${record.response?.model?.version || "unknown"}`,
      "Probabilistic result, not proof"
    ].join(". ");
    box.setAttribute("aria-label", details);
    box.title = details;
    root.append(box);
    return box;
  }

  function createBadge(rect, text, tone) {
    const root = ensureOverlayRoot();
    const badge = document.createElement("div");
    badge.className = `dfd-image-badge ${tone}`;
    badge.textContent = text;
    badge.style.left = `${Math.max(4, rect.left)}px`;
    badge.style.top = `${Math.max(4, rect.top)}px`;
    badge.style.maxWidth = `${Math.max(120, Math.min(rect.width, 280))}px`;
    root.append(badge);
    return badge;
  }

  function removeRecordOverlays(record) {
    for (const node of record.overlayNodes) {
      node.remove();
    }
    record.overlayNodes = [];
  }

  function scheduleReposition() {
    if (state.repositionFrame) {
      return;
    }
    state.repositionFrame = requestAnimationFrame(() => {
      state.repositionFrame = 0;
      for (const record of state.recordsById.values()) {
        if (record.overlayNodes.length || record.response) {
          renderRecordOverlay(record);
        }
      }
      updateDiagnosticsPanel();
    });
  }

  function attachRepositionListeners() {
    if (state.listenersAttached) {
      return;
    }
    window.addEventListener("scroll", scheduleReposition, true);
    window.addEventListener("resize", scheduleReposition);
    state.listenersAttached = true;
  }

  function resetRecordsForNewSession() {
    for (const record of state.recordsById.values()) {
      removeRecordOverlays(record);
      record.state = IMAGE_SCAN_STATE.notScanned;
      record.sourceKey = "";
      record.submittedSignature = "";
      record.response = null;
      record.error = null;
      record.cacheHit = false;
      record.isNearViewport = false;
      if (state.intersectionObserver) {
        state.intersectionObserver.observe(record.img);
      }
      evaluateImage(record.img, "new_session");
    }
  }

  function ensureOverlayRoot() {
    let root = document.getElementById(OVERLAY_ROOT_ID);
    if (root) {
      return root;
    }
    root = document.createElement("div");
    root.id = OVERLAY_ROOT_ID;
    root.setAttribute("aria-hidden", "true");
    document.documentElement.append(root);
    return root;
  }

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) {
      return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = [
      `#${OVERLAY_ROOT_ID}{position:fixed;inset:0;z-index:2147483646;pointer-events:none;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#111827;}`,
      ".dfd-image-badge{position:fixed;box-sizing:border-box;padding:4px 6px;border:1px solid rgba(17,24,39,.35);border-radius:4px;background:#fff;color:#111827;font-size:12px;line-height:1.25;font-weight:700;box-shadow:0 6px 18px rgba(0,0,0,.18);overflow-wrap:anywhere;}",
      ".dfd-image-badge.running{border-color:#6b7280;background:#f9fafb;}",
      ".dfd-image-badge.completed{border-color:#1d4ed8;background:#eff6ff;color:#1e3a8a;}",
      ".dfd-image-badge.neutral{border-color:#6b7280;background:#f3f4f6;color:#374151;}",
      ".dfd-image-badge.failed{border-color:#991b1b;background:#fee2e2;color:#7f1d1d;}",
      ".dfd-face-box{position:fixed;box-sizing:border-box;border:2px solid #374151;border-radius:4px;background:rgba(255,255,255,.08);box-shadow:0 0 0 1px rgba(255,255,255,.8);}",
      ".dfd-face-box.fake{border-color:#b91c1c;}",
      ".dfd-face-box.real{border-color:#047857;}",
      ".dfd-face-box.uncertain{border-color:#a16207;}",
      ".dfd-face-label{position:absolute;left:-2px;top:-24px;max-width:220px;padding:3px 5px;border-radius:4px;background:#111827;color:#fff;font-size:11px;font-weight:700;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}",
      ".dfd-face-label.inside{top:0;}",
      "#dfd-diagnostics-panel{position:fixed;left:8px;bottom:8px;z-index:2147483647;pointer-events:none;max-width:300px;padding:8px;border:1px solid rgba(17,24,39,.25);border-radius:4px;background:#fff;color:#111827;font:12px/1.35 system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;box-shadow:0 6px 18px rgba(0,0,0,.18);}",
      "#dfd-diagnostics-panel strong{display:block;margin-bottom:4px;}"
    ].join("");
    document.documentElement.append(style);
  }

  function updateDiagnosticsPanel() {
    const existing = document.getElementById("dfd-diagnostics-panel");
    if (!state.options.diagnosticsEnabled) {
      if (existing) {
        existing.remove();
      }
      return;
    }
    const panel = existing || document.createElement("div");
    panel.id = "dfd-diagnostics-panel";
    panel.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = "Deepfake scan diagnostics";
    panel.append(title);
    const lines = [
      `Discovered: ${state.diagnostics.imagesDiscovered}`,
      `Eligible: ${state.diagnostics.imagesEligible}`,
      `Skipped: ${state.diagnostics.imagesSkipped}`,
      `Queued: ${state.diagnostics.queuedRequests}`,
      `Cache hits: ${state.diagnostics.cacheHits}`,
      `Cache misses: ${state.diagnostics.cacheMisses}`,
      `Completed: ${state.diagnostics.completedAnalyses}`,
      `No face: ${state.diagnostics.noFaceResults}`,
      `Failed: ${state.diagnostics.failedAnalyses}`,
      `Average API: ${averageApiTime()} ms`,
      `Model: ${activeModelLabel()}`
    ];
    for (const line of lines) {
      const div = document.createElement("div");
      div.textContent = line;
      panel.append(div);
    }
    if (!existing) {
      document.documentElement.append(panel);
    }
  }

  function showToast(text, toastState) {
    const toast = getOrCreateToast();
    toast.textContent = String(text);
    toast.dataset.state = String(toastState);
    clearTimeout(showToast.removeTimer);
    showToast.removeTimer = setTimeout(() => {
      const current = document.getElementById(TOAST_ID);
      if (current) {
        current.remove();
      }
    }, REMOVE_DELAY_MS);
  }

  function getOrCreateToast() {
    let toast = document.getElementById(TOAST_ID);
    if (toast) {
      return toast;
    }
    toast = document.createElement("div");
    toast.id = TOAST_ID;
    toast.setAttribute("role", "status");
    toast.style.position = "fixed";
    toast.style.right = "16px";
    toast.style.bottom = "16px";
    toast.style.zIndex = "2147483647";
    toast.style.maxWidth = "320px";
    toast.style.padding = "10px 12px";
    toast.style.border = "1px solid rgba(20, 28, 38, 0.24)";
    toast.style.borderRadius = "6px";
    toast.style.background = "#ffffff";
    toast.style.color = "#18202b";
    toast.style.boxShadow = "0 8px 28px rgba(0, 0, 0, 0.18)";
    toast.style.font = "13px/1.4 system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    document.documentElement.appendChild(toast);
    return toast;
  }

  function createDiagnostics() {
    return {
      imagesDiscovered: 0,
      imagesEligible: 0,
      imagesSkipped: 0,
      queuedRequests: 0,
      cacheHits: 0,
      cacheMisses: 0,
      completedAnalyses: 0,
      noFaceResults: 0,
      failedAnalyses: 0,
      apiTimeTotal: 0,
      apiTimeCount: 0
    };
  }

  function activeModelLabel() {
    const model = state.backendMetadata?.model;
    return model ? `${model.id}@${model.version}` : "unknown";
  }

  function averageApiTime() {
    if (!state.diagnostics.apiTimeCount) {
      return "0.0";
    }
    return (state.diagnostics.apiTimeTotal / state.diagnostics.apiTimeCount).toFixed(1);
  }

  function statusText(value) {
    if (value === IMAGE_SCAN_STATE.downloading) {
      return "Downloading";
    }
    if (value === IMAGE_SCAN_STATE.analysing) {
      return "Analysing";
    }
    return "Queued";
  }

  function faceLabelText(face, response) {
    const index = Number(face.face_index) + 1;
    const score = state.options.showModelScores ? ` · Fake ${formatPercent(face.fake_score)}` : "";
    const mock = response?.mock ? " · Mock" : "";
    return `Face ${index}: ${face.label}${score}${mock}`;
  }

  function labelClass(label) {
    if (label === "Likely Fake") {
      return "fake";
    }
    if (label === "Likely Real") {
      return "real";
    }
    return "uncertain";
  }

  function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "n/a";
  }

  function formatScore(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(4) : "n/a";
  }

  function isElementVisible(img, rect) {
    const style = getComputedStyle(img);
    return (
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      Number(style.opacity || "1") > 0 &&
      rect.width > 0 &&
      rect.height > 0 &&
      img.isConnected
    );
  }

  function isHttpImageUrl(value) {
    try {
      const url = new URL(value || "");
      return url.protocol === "http:" || url.protocol === "https:";
    } catch {
      return false;
    }
  }

  function hasExtremeAspectRatio(width, height) {
    const ratio = Number(width) / Number(height);
    return Number.isFinite(ratio) && (ratio > 4 || ratio < 0.25);
  }

  function isLikelyIconOrLogo(value, naturalWidth, naturalHeight) {
    if (!/(^|[-_/])(favicon|icon|logo|sprite|badge|tracking|pixel)([-_.?/]|$)/i.test(value || "")) {
      return false;
    }
    return Math.max(Number(naturalWidth) || 0, Number(naturalHeight) || 0) <= 512;
  }

  function computeRenderedImageGeometry({ naturalWidth, naturalHeight, displayedWidth, displayedHeight, objectFit, objectPosition }) {
    const natural = { width: positiveNumber(naturalWidth), height: positiveNumber(naturalHeight) };
    const displayed = { width: positiveNumber(displayedWidth), height: positiveNumber(displayedHeight) };
    if (!natural.width || !natural.height || !displayed.width || !displayed.height) {
      return { valid: false };
    }
    const fit = ["fill", "contain", "cover", "none", "scale-down"].includes(objectFit) ? objectFit : "fill";
    const concrete = concreteObjectSize(fit, natural, displayed);
    const position = parseObjectPosition(objectPosition);
    const offsetX = computeObjectOffset(displayed.width - concrete.width, position.x);
    const offsetY = computeObjectOffset(displayed.height - concrete.height, position.y);
    return {
      valid: true,
      displayedWidth: displayed.width,
      displayedHeight: displayed.height,
      scaleX: concrete.width / natural.width,
      scaleY: concrete.height / natural.height,
      offsetX,
      offsetY
    };
  }

  function mapFaceBoxToDisplayedImage(box, geometry) {
    if (!geometry?.valid || !box) {
      return { visible: false };
    }
    const raw = {
      x1: geometry.offsetX + Number(box.x1) * geometry.scaleX,
      y1: geometry.offsetY + Number(box.y1) * geometry.scaleY,
      x2: geometry.offsetX + Number(box.x2) * geometry.scaleX,
      y2: geometry.offsetY + Number(box.y2) * geometry.scaleY
    };
    const clipped = {
      x1: clamp(raw.x1, 0, geometry.displayedWidth),
      y1: clamp(raw.y1, 0, geometry.displayedHeight),
      x2: clamp(raw.x2, 0, geometry.displayedWidth),
      y2: clamp(raw.y2, 0, geometry.displayedHeight)
    };
    const width = Math.max(0, clipped.x2 - clipped.x1);
    const height = Math.max(0, clipped.y2 - clipped.y1);
    return {
      visible: width > 0 && height > 0,
      x: clipped.x1,
      y: clipped.y1,
      width,
      height
    };
  }

  function concreteObjectSize(fit, natural, displayed) {
    if (fit === "fill") {
      return { width: displayed.width, height: displayed.height };
    }
    if (fit === "none") {
      return { width: natural.width, height: natural.height };
    }
    if (fit === "scale-down") {
      if (natural.width <= displayed.width && natural.height <= displayed.height) {
        return { width: natural.width, height: natural.height };
      }
      return containedSize(natural, displayed);
    }
    if (fit === "cover") {
      const scale = Math.max(displayed.width / natural.width, displayed.height / natural.height);
      return { width: natural.width * scale, height: natural.height * scale };
    }
    return containedSize(natural, displayed);
  }

  function containedSize(natural, displayed) {
    const scale = Math.min(displayed.width / natural.width, displayed.height / natural.height);
    return { width: natural.width * scale, height: natural.height * scale };
  }

  function parseObjectPosition(value) {
    const tokens = String(value || "50% 50%").trim().split(/\s+/).filter(Boolean);
    if (tokens.length === 1) {
      if (tokens[0] === "top" || tokens[0] === "bottom") {
        return { x: percent(0.5), y: parsePositionToken(tokens[0], "y") };
      }
      return { x: parsePositionToken(tokens[0], "x"), y: percent(0.5) };
    }
    return {
      x: parsePositionToken(tokens[0] || "50%", "x"),
      y: parsePositionToken(tokens[1] || "50%", "y")
    };
  }

  function parsePositionToken(token, axis) {
    const lower = String(token || "").toLowerCase();
    if (lower === "center") {
      return percent(0.5);
    }
    if ((axis === "x" && lower === "left") || (axis === "y" && lower === "top")) {
      return percent(0);
    }
    if ((axis === "x" && lower === "right") || (axis === "y" && lower === "bottom")) {
      return percent(1);
    }
    if (lower.endsWith("%")) {
      const parsed = Number.parseFloat(lower.slice(0, -1));
      return Number.isFinite(parsed) ? percent(parsed / 100) : percent(0.5);
    }
    if (lower.endsWith("px")) {
      const parsed = Number.parseFloat(lower.slice(0, -2));
      return Number.isFinite(parsed) ? { type: "px", value: parsed } : percent(0.5);
    }
    return percent(0.5);
  }

  function computeObjectOffset(extraSpace, position) {
    return position.type === "px" ? position.value : extraSpace * position.value;
  }

  function percent(value) {
    return { type: "percent", value };
  }

  function positiveNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : 0;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }
})();
