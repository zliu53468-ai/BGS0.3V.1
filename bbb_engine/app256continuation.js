(() => {
"use strict";

const BASE = (typeof window !== "undefined") ? window.__BGS256_TEST__ : null;
if (!BASE) return;

const VERSION = "V23_SHORT_X_DYNAMIC_HAZARD_R1";
const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";
const PROB_MIN = 0.42;
const PROB_MAX = 0.58;
const GAP_SCALE = 0.30;

const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const signed = v => clip(v, -1, 1);
const bp = seq => seq.filter(x => x === "B" || x === "P");
const sideSign = side => side === "B" ? 1 : side === "P" ? -1 : 0;

function transitionSequence(seq) {
  const a = bp(seq), out = [];
  for (let i = 1; i < a.length; i++) out.push(a[i] === a[i - 1] ? "S" : "X");
  return out;
}

function transitionMotifOrder(tokens, order) {
  if (order < 1 || tokens.length <= order) return null;
  const context = tokens.slice(-order).join("");
  let same = 0, sw = 0, total = 0;
  const start = Math.max(0, tokens.length - 120);
  for (let i = start; i + order < tokens.length; i++) {
    if (tokens.slice(i, i + order).join("") !== context) continue;
    const age = tokens.length - 1 - (i + order);
    const w = Math.pow(0.965, age);
    total += w;
    if (tokens[i + order] === "S") same += w;
    else sw += w;
  }
  if (total <= 0) return null;
  const prior = 0.62, denom = total + 2 * prior;
  return {
    order,
    context,
    pSame: clip((same + prior) / denom),
    pSwitch: clip((sw + prior) / denom),
    support: clip(total / 3.2),
    weightedSamples: total
  };
}

function transitionMotifBackoff(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, order: 0, agreement: 0, details: [] };
  const details = [];
  for (let order = Math.min(5, tokens.length); order >= 1; order--) {
    const item = transitionMotifOrder(tokens, order);
    if (item) details.push(item);
  }
  if (!details.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, order: 0, agreement: 0, details: [] };

  const orderWeights = { 5: 1.00, 4: 0.84, 3: 0.68, 2: 0.52, 1: 0.36 };
  let sameSum = 0, weightSum = 0, supportSum = 0;
  const dirs = [];
  for (const item of details) {
    const ow = orderWeights[item.order] || 0.30;
    const w = ow * (0.35 + 0.65 * item.support);
    sameSum += item.pSame * w;
    weightSum += w;
    supportSum += ow * item.support;
    if (Math.abs(item.pSame - 0.5) >= 0.04) dirs.push(Math.sign(item.pSame - 0.5));
  }
  const raw = weightSum ? sameSum / weightSum : 0.5;
  const support = clip(supportSum / 2.5);
  const agreement = dirs.length ? Math.abs(dirs.reduce((a, b) => a + b, 0)) / dirs.length : 0;
  const shrink = 0.48 + 0.52 * support;
  const pSame = clip(0.5 + (raw - 0.5) * shrink, 0.12, 0.88);
  return { pSame, pSwitch: 1 - pSame, support, order: details[0]?.order || 0, agreement, details };
}

function transitionDepthForecast(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, token: "", depth: 0, pPersist: 0.5 };
  const token = tokens.at(-1);
  let depth = 1;
  for (let i = tokens.length - 2; i >= 0 && tokens[i] === token; i--) depth++;

  const completedRuns = [];
  let current = tokens[0], n = 1;
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === current) n++;
    else { completedRuns.push({ token: current, length: n }); current = tokens[i]; n = 1; }
  }

  const items = completedRuns.filter(r => r.token === token).slice(-24);
  let reached = 0, persisted = 0, ended = 0;
  for (let i = 0; i < items.length; i++) {
    if (items[i].length < depth) continue;
    const w = Math.pow(0.95, items.length - 1 - i);
    reached += w;
    if (items[i].length > depth) persisted += w;
    else ended += w;
  }
  const prior = 0.70, denom = reached + 2 * prior;
  const rawPersist = denom > 0 ? clip((persisted + prior) / denom) : 0.5;
  const support = clip(reached / 4);
  const pPersist = clip(0.5 + (rawPersist - 0.5) * (0.42 + 0.58 * support), 0.12, 0.88);
  const pSame = token === "S" ? pPersist : 1 - pPersist;
  return { pSame, pSwitch: 1 - pSame, support, token, depth, pPersist, reached, persisted, ended };
}

function currentRoadState(seq) {
  const road = BASE.buildBigRoad(seq);
  const current = road.currentStreak || null;
  const completed = road.streaks.length > 1 ? road.streaks.slice(0, -1) : [];
  const side = current?.side || "";
  const length = current?.logicalLength || 0;
  const opposite = side === "B" ? "P" : "B";
  return { road, current, completed, side, length, opposite, sign: sideSign(side) };
}

function stageCurve(completed, side, maxStage = 6) {
  const curve = [];
  for (let stage = 1; stage <= maxStage; stage++) {
    const s = BASE.stageSurvival(completed, side, stage);
    curve.push({ stage, pContinue: s.cont, pTurn: s.turn, support: s.support, reached: s.reached || 0 });
  }
  return curve;
}

function conditionalStageEvidence(completed, side, length) {
  if (!side || !length) return {
    pSame: 0.5, support: 0, contextPSame: 0.5, contextSupport: 0,
    nextPSame: 0.5, nextSupport: 0, cliff: 0, cliffSupport: 0, curve: []
  };
  const now = BASE.stageSurvival(completed, side, length);
  const next = BASE.stageSurvival(completed, side, length + 1);
  const context = BASE.contextualStageStats(completed, completed.length, side, length);
  const curve = stageCurve(completed, side, Math.max(6, Math.min(8, length + 2)));
  const cliff = clip(now.cont - next.cont, 0, 1);
  const cliffSupport = clip(Math.min(now.support || 0, Math.max(0.20, next.support || 0)));
  return {
    pSame: now.cont,
    pSwitch: now.turn,
    support: now.support || 0,
    contextPSame: context.cont,
    contextPSwitch: context.turn,
    contextSupport: context.support || 0,
    nextPSame: next.cont,
    nextSupport: next.support || 0,
    cliff,
    cliffSupport,
    curve
  };
}

function candidateEvidence(seq, state, basePrediction) {
  const cand = basePrediction?.candidates || BASE.bigRoadCandidates(seq);
  if (!state.side) return { pSame: 0.5, support: 0, sameScore: 0.5, switchScore: 0.5 };
  const sameScore = clip(cand?.[state.side] ?? 0.5);
  const switchScore = clip(cand?.[state.opposite] ?? 0.5);
  const diff = sameScore - switchScore;
  const pSame = clip(1 / (1 + Math.exp(-diff / 0.12)), 0.18, 0.82);
  return { pSame, pSwitch: 1 - pSame, support: clip(cand?.support || 0), sameScore, switchScore, diff };
}

function baseBackgroundEvidence(basePrediction, state) {
  if (!state.side) return { pSame: 0.5, support: 0 };
  const gap = Number(basePrediction?.gap || 0);
  const sideGap = state.sign * gap;
  const pSame = clip(1 / (1 + Math.exp(-sideGap / 0.42)), 0.32, 0.68);
  const support = clip(Math.abs(gap) / 0.24);
  return { pSame, pSwitch: 1 - pSame, support, gap, sideGap };
}

function singleHazardSignals(seq, basePrediction) {
  const state = currentRoadState(seq);
  if (!state.side) return {
    state, pSame: 0.5, pSwitch: 0.5, support: 0, sameEdge: 0,
    motif: transitionMotifBackoff(seq), depth: transitionDepthForecast(seq)
  };

  const motif = transitionMotifBackoff(seq);
  const depth = transitionDepthForecast(seq);
  const stage = conditionalStageEvidence(state.completed, state.side, state.length);
  const candidate = candidateEvidence(seq, state, basePrediction);
  const background = baseBackgroundEvidence(basePrediction, state);

  const shortSwitchPhase = state.length === 1 && depth.token === "X";
  const switchConsensus = shortSwitchPhase && motif.pSwitch >= 0.54 && depth.pSwitch >= 0.54;
  const consensusSupport = Math.sqrt(clip(motif.support) * clip(depth.support));
  const consensusStrengthRaw = switchConsensus
    ? clip(((motif.pSwitch - 0.50) + (depth.pSwitch - 0.50)) / 0.28)
    : 0;
  const consensusStrength = consensusStrengthRaw * consensusSupport;
  const staleScale = switchConsensus ? clip(1 - 0.58 * consensusStrength, 0.42, 1) : 1;

  const stageBase = shortSwitchPhase ? 0.17 : 0.30;
  const contextBase = shortSwitchPhase ? 0.12 : 0.20;
  const motifBase = shortSwitchPhase ? 0.28 : 0.20;
  const depthBase = shortSwitchPhase ? 0.22 : 0.14;
  const candidateBase = shortSwitchPhase ? 0.07 : 0.09;

  const stageWeight = stageBase * (0.28 + 0.72 * stage.support) * staleScale;
  const contextWeight = contextBase * (0.28 + 0.72 * stage.contextSupport) * staleScale;
  const motifWeight = motifBase * (0.30 + 0.70 * motif.support) * (0.60 + 0.40 * motif.agreement);
  const depthWeight = depthBase * (0.30 + 0.70 * depth.support);
  const candidateWeight = candidateBase * (0.30 + 0.70 * candidate.support);

  const hazardSupport = shortSwitchPhase
    ? clip(
        0.18 * stage.support +
        0.12 * stage.contextSupport +
        0.30 * motif.support +
        0.24 * depth.support +
        0.16 * candidate.support
      )
    : clip(
        0.30 * stage.support +
        0.20 * stage.contextSupport +
        0.22 * motif.support +
        0.14 * depth.support +
        0.14 * candidate.support
      );

  const backgroundWeight = shortSwitchPhase
    ? 0.16 * (1 - 0.72 * hazardSupport) + 0.03
    : 0.22 * (1 - 0.68 * hazardSupport) + 0.04;
  let neutralWeight = 0.34 * (1 - 0.55 * hazardSupport);
  if (shortSwitchPhase) neutralWeight *= 0.72;

  let numerator = 0.5 * neutralWeight;
  let denominator = neutralWeight;
  const add = (p, w) => { numerator += clip(p) * w; denominator += w; };
  add(stage.pSame, stageWeight);
  add(stage.contextPSame, contextWeight);
  add(motif.pSame, motifWeight);
  add(depth.pSame, depthWeight);
  add(candidate.pSame, candidateWeight);
  add(background.pSame, backgroundWeight);

  let pSame = denominator > 0 ? numerator / denominator : 0.5;

  const cliffGate = state.length >= 3 ? clip((state.length - 2) / 2) : 0;
  const cliffEvidence = stage.cliff * stage.cliffSupport * cliffGate;
  pSame -= 0.22 * cliffEvidence;

  let shortSwitchBoost = 0;
  if (switchConsensus) {
    const evidence = clip(
      0.52 * clip((motif.pSwitch - 0.50) / 0.22) * (0.35 + 0.65 * motif.support) +
      0.48 * clip((depth.pSwitch - 0.50) / 0.22) * (0.35 + 0.65 * depth.support)
    );
    shortSwitchBoost = 0.045 * evidence * (0.25 + 0.75 * consensusSupport);
    pSame -= shortSwitchBoost;
  }

  let formationBoost = 0;
  if (state.length === 2) {
    const stageEdge = stage.pSame - 0.5;
    const contextEdge = stage.contextPSame - 0.5;
    const aligned = stageEdge * contextEdge > 0;
    const agreementEdge = aligned
      ? Math.sign(stageEdge) * Math.min(Math.abs(stageEdge), Math.abs(contextEdge))
      : 0;
    const formationSupport = Math.min(1, 0.55 * stage.support + 0.45 * stage.contextSupport);
    formationBoost = signed(agreementEdge / 0.32) * formationSupport * 0.045;
    pSame += formationBoost;
  }

  pSame = clip(pSame, 0.16, 0.84);
  const sameEdge = signed((pSame - 0.5) * 2);
  const support = clip(0.72 * hazardSupport + 0.18 * background.support + 0.10 * Math.abs(sameEdge));

  return {
    state,
    pSame,
    pSwitch: 1 - pSame,
    sameEdge,
    support,
    hazardSupport,
    motif,
    depth,
    stage,
    candidate,
    background,
    cliffEvidence,
    formationBoost,
    shortSwitchPhase,
    switchConsensus,
    consensusSupport,
    consensusStrengthRaw,
    consensusStrength,
    staleScale,
    shortSwitchBoost,
    weights: { stageWeight, contextWeight, motifWeight, depthWeight, candidateWeight, backgroundWeight, neutralWeight }
  };
}

function hazardChoose(seq) {
  const base = BASE.choose(seq);
  const sig = singleHazardSignals(seq, base);
  const currentSide = sig.state.side;
  if (!currentSide) return base;

  const direction = sig.pSame >= 0.5 ? currentSide : sig.state.opposite;
  const pBUnclipped = currentSide === "B" ? sig.pSame : 1 - sig.pSame;
  const pB = clip(pBUnclipped, PROB_MIN, PROB_MAX), pP = 1 - pB;
  const confidence = direction === "B" ? pB : pP;
  const gap = sig.state.sign * sig.sameEdge * GAP_SCALE;

  let regime = "S/X平衡";
  if (sig.shortSwitchPhase && sig.switchConsensus && sig.pSame < 0.50) regime = "短交錯切換";
  else if (sig.state.length >= 3 && sig.cliffEvidence >= 0.08 && sig.pSame < 0.50) regime = "條件斷點";
  else if (sig.pSame >= 0.56) regime = "條件延續";
  else if (sig.pSame <= 0.44) regime = "條件切換";

  const strength = clip(0.42 + 0.36 * sig.support + 0.18 * Math.abs(sig.sameEdge) + 0.04 * Math.min(1, sig.state.length / 4));

  const diag = {
    version: VERSION,
    baseGap: base.gap,
    finalGap: gap,
    pSame: sig.pSame,
    pSwitch: sig.pSwitch,
    currentSide,
    currentLength: sig.state.length,
    transitionToken: sig.depth.token,
    transitionDepth: sig.depth.depth,
    motifPSame: sig.motif.pSame,
    motifSupport: sig.motif.support,
    motifAgreement: sig.motif.agreement,
    depthPSame: sig.depth.pSame,
    depthSupport: sig.depth.support,
    stagePSame: sig.stage.pSame,
    nextStagePSame: sig.stage.nextPSame,
    stageSupport: sig.stage.support,
    contextPSame: sig.stage.contextPSame,
    contextSupport: sig.stage.contextSupport,
    survivalCliff: sig.stage.cliff,
    cliffEvidence: sig.cliffEvidence,
    formationBoost: sig.formationBoost,
    hazardSupport: sig.hazardSupport,
    backgroundWeight: sig.weights.backgroundWeight,
    shortSwitchPhase: sig.shortSwitchPhase,
    switchConsensus: sig.switchConsensus,
    consensusSupport: sig.consensusSupport,
    consensusStrengthRaw: sig.consensusStrengthRaw,
    consensusStrength: sig.consensusStrength,
    staleScale: sig.staleScale,
    shortSwitchBoost: sig.shortSwitchBoost
  };

  return {
    ...base,
    direction,
    gap,
    confidence,
    probabilities: { B: pB, P: pP },
    regime,
    strength,
    singleHazard: sig,
    v22: diag,
    v23: diag
  };
}

function continuationSignals(seq, basePrediction) { return singleHazardSignals(seq, basePrediction || BASE.choose(seq)); }
function enhancedChoose(seq) { return hazardChoose(seq); }
function expectedBranchFutureQuality(seq, first) {
  const road = BASE.buildBigRoad([...seq, first]);
  const current = road.currentStreak;
  if (!current) return { quality: 0.5, pSame: 0.5, support: 0 };
  const completed = road.streaks.length > 1 ? road.streaks.slice(0, -1) : [];
  const stage = BASE.stageSurvival(completed, first, current.logicalLength);
  const context = BASE.contextualStageStats(completed, completed.length, first, current.logicalLength);
  const support = clip(0.56 * (stage.support || 0) + 0.44 * (context.support || 0));
  const pSame = clip((0.56 * stage.cont + 0.44 * context.cont), 0.12, 0.88);
  return { quality: pSame, pSame, support };
}

function readHistory() {
  if (typeof localStorage === "undefined") return [];
  for (const key of [
    "bgs256d_frozen_6x15_forward_v18",
    "bgs256d_frozen_6x15_sensitive_v17",
    "bgs256d_frozen_6x15_bigroad_v16"
  ]) {
    try {
      const raw = JSON.parse(localStorage.getItem(key) || "null");
      if (raw && Array.isArray(raw.history)) return raw.history.filter(x => ["B", "P", "T"].includes(x)).slice(-500);
    } catch (_) {}
  }
  return [];
}

function saveSelection(direction) {
  if (typeof localStorage === "undefined") return;
  try {
    const old = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || {};
    const streak = old.last_selected === direction ? Math.max(1, (+old.selection_streak || 0) + 1) : 1;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ last_selected: direction, selection_streak: streak }));
  } catch (_) {}
}

function renderPrediction(p, historyLength) {
  const el = id => document.getElementById(id);
  const orb = el("directionOrb");
  if (!orb) return;
  const isB = p.direction === "B";
  el("directionText").textContent = isB ? "莊" : "閒";
  el("directionCode").textContent = isB ? "BANKER" : "PLAYER";
  el("confidence").textContent = (p.confidence * 100).toFixed(1) + "%";
  el("regime").textContent = p.regime;
  el("strength").textContent = p.strength >= .68 ? "穩定" : p.strength >= .52 ? "中等" : "保守";
  orb.className = "direction-orb " + (isB ? "banker" : "player");
  if (el("modePill")) el("modePill").textContent = "分析完成";
  if (el("roundCount")) el("roundCount").textContent = historyLength;
  if (el("message")) el("message").textContent = `第 ${historyLength + 1} 局分析完成`;
}

function installUIOverride() {
  if (typeof document === "undefined") return;
  const oldBtn = document.getElementById("btnStart");
  if (!oldBtn) return;
  const btn = oldBtn.cloneNode(true);
  oldBtn.replaceWith(btn);
  btn.addEventListener("click", () => {
    const history = readHistory();
    if (!history.length) {
      const msg = document.getElementById("message");
      if (msg) { msg.textContent = "請先輸入牌局紀錄"; msg.classList.add("warning"); }
      return;
    }
    const p = hazardChoose(history);
    saveSelection(p.direction);
    renderPrediction(p, history.length);
  });

  const end = document.getElementById("btnEnd");
  if (end) end.addEventListener("click", () => {
    try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
  });
}

if (typeof window !== "undefined") {
  window.__BGS256_CONTINUATION_TEST__ = {
    version: VERSION,
    transitionSequence,
    transitionMotifBackoff,
    transitionDepthForecast,
    conditionalStageEvidence,
    singleHazardSignals,
    continuationSignals,
    hazardChoose,
    enhancedChoose,
    expectedBranchFutureQuality
  };
}

installUIOverride();
})();