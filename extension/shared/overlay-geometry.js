export function computeRenderedImageGeometry({
  naturalWidth,
  naturalHeight,
  displayedWidth,
  displayedHeight,
  objectFit = "fill",
  objectPosition = "50% 50%"
}) {
  const natural = {
    width: positiveNumber(naturalWidth),
    height: positiveNumber(naturalHeight)
  };
  const displayed = {
    width: positiveNumber(displayedWidth),
    height: positiveNumber(displayedHeight)
  };
  if (!natural.width || !natural.height || !displayed.width || !displayed.height) {
    return emptyGeometry();
  }

  const fit = normalizeObjectFit(objectFit);
  const concrete = concreteObjectSize(fit, natural, displayed);
  const position = parseObjectPosition(objectPosition);
  const offsetX = computeObjectOffset(displayed.width - concrete.width, position.x);
  const offsetY = computeObjectOffset(displayed.height - concrete.height, position.y);

  return {
    valid: true,
    naturalWidth: natural.width,
    naturalHeight: natural.height,
    displayedWidth: displayed.width,
    displayedHeight: displayed.height,
    objectFit: fit,
    objectPosition: String(objectPosition || "50% 50%"),
    contentRect: {
      x: offsetX,
      y: offsetY,
      width: concrete.width,
      height: concrete.height
    },
    scaleX: concrete.width / natural.width,
    scaleY: concrete.height / natural.height,
    offsetX,
    offsetY
  };
}

export function mapFaceBoxToDisplayedImage(box, geometry) {
  if (!geometry?.valid || !box) {
    return invisibleBox();
  }
  const raw = {
    x1: geometry.offsetX + Number(box.x1) * geometry.scaleX,
    y1: geometry.offsetY + Number(box.y1) * geometry.scaleY,
    x2: geometry.offsetX + Number(box.x2) * geometry.scaleX,
    y2: geometry.offsetY + Number(box.y2) * geometry.scaleY
  };
  if (!Object.values(raw).every(Number.isFinite)) {
    return invisibleBox();
  }

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
    clipped:
      clipped.x1 !== raw.x1 ||
      clipped.y1 !== raw.y1 ||
      clipped.x2 !== raw.x2 ||
      clipped.y2 !== raw.y2,
    x: clipped.x1,
    y: clipped.y1,
    width,
    height,
    raw
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
  if (tokens.length === 0) {
    return { x: percent(0.5), y: percent(0.5) };
  }
  if (tokens.length === 1) {
    const only = parsePositionToken(tokens[0], "x");
    if (tokens[0] === "top" || tokens[0] === "bottom") {
      return { x: percent(0.5), y: parsePositionToken(tokens[0], "y") };
    }
    return { x: only, y: percent(0.5) };
  }
  return {
    x: parsePositionToken(tokens[0], "x"),
    y: parsePositionToken(tokens[1], "y")
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
    const value = Number.parseFloat(lower.slice(0, -1));
    return Number.isFinite(value) ? percent(value / 100) : percent(0.5);
  }
  if (lower.endsWith("px")) {
    const value = Number.parseFloat(lower.slice(0, -2));
    return Number.isFinite(value) ? { type: "px", value } : percent(0.5);
  }
  return percent(0.5);
}

function computeObjectOffset(extraSpace, position) {
  if (position.type === "px") {
    return position.value;
  }
  return extraSpace * position.value;
}

function percent(value) {
  return { type: "percent", value };
}

function normalizeObjectFit(value) {
  const fit = String(value || "fill").trim();
  return ["fill", "contain", "cover", "none", "scale-down"].includes(fit) ? fit : "fill";
}

function emptyGeometry() {
  return {
    valid: false,
    contentRect: { x: 0, y: 0, width: 0, height: 0 },
    scaleX: 0,
    scaleY: 0,
    offsetX: 0,
    offsetY: 0
  };
}

function invisibleBox() {
  return { visible: false, clipped: false, x: 0, y: 0, width: 0, height: 0, raw: null };
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
