// Texas Hold'em: names your best hand and estimates how often it wins
// against a number of opponents by dealing the unknown cards many times.
import { parse, fullDeck, showAll } from '../cards.js';

export const HAND_NAMES = [
  'Vysoká karta', 'Pár', 'Dva páry', 'Trojica', 'Postupka',
  'Farba', 'Full house', 'Štvorica', 'Postupka vo farbe',
];

// Highest straight in a 15-bit rank mask (bit r set = rank r present), 0 if none.
function straightTop(mask) {
  if (mask & (1 << 14)) mask |= 1 << 1; // the ace also plays low: A-2-3-4-5
  for (let top = 14; top >= 5; top--) {
    const run = 0b11111 << (top - 4);
    if ((mask & run) === run) return top;
  }
  return 0;
}

// Score of the best 5-card hand from 5–7 cards [{rank, suit}]. Higher is better;
// Math.floor(score / 16 ** 5) is the index into HAND_NAMES.
export function score(cards) {
  const counts = new Array(15).fill(0);
  const bySuit = { S: 0, H: 0, D: 0, C: 0 };
  const suitCount = { S: 0, H: 0, D: 0, C: 0 };
  let mask = 0;
  for (const c of cards) {
    counts[c.rank]++;
    mask |= 1 << c.rank;
    bySuit[c.suit] |= 1 << c.rank;
    suitCount[c.suit]++;
  }
  const pack = (cat, ...ranks) => {
    let s = cat;
    for (let i = 0; i < 5; i++) s = s * 16 + (ranks[i] ?? 0);
    return s;
  };

  let flushMask = 0;
  for (const s in suitCount) if (suitCount[s] >= 5) flushMask = bySuit[s];
  if (flushMask) {
    const sf = straightTop(flushMask);
    if (sf) return pack(8, sf);
  }

  const quads = [], trips = [], pairs = [], singles = [];
  for (let r = 14; r >= 2; r--) {
    if (counts[r] === 4) quads.push(r);
    else if (counts[r] === 3) trips.push(r);
    else if (counts[r] === 2) pairs.push(r);
    else if (counts[r] === 1) singles.push(r);
  }
  const highest = (exclude, n) => {
    const out = [];
    for (let r = 14; r >= 2 && out.length < n; r--) if (counts[r] && !exclude.includes(r)) out.push(r);
    return out;
  };

  if (quads.length) return pack(7, quads[0], ...highest([quads[0]], 1));
  if (trips.length && (trips.length > 1 || pairs.length)) {
    const pair = Math.max(trips[1] ?? 0, pairs[0] ?? 0);
    return pack(6, trips[0], pair);
  }
  if (flushMask) {
    const top = [];
    for (let r = 14; r >= 2 && top.length < 5; r--) if (flushMask & (1 << r)) top.push(r);
    return pack(5, ...top);
  }
  const st = straightTop(mask);
  if (st) return pack(4, st);
  if (trips.length) return pack(3, trips[0], ...highest([trips[0]], 2));
  if (pairs.length >= 2) return pack(2, pairs[0], pairs[1], ...highest([pairs[0], pairs[1]], 1));
  if (pairs.length) return pack(1, pairs[0], ...highest([pairs[0]], 3));
  return pack(0, ...highest([], 5));
}

export function handName(codes) {
  const s = score(codes.map(parse));
  const cat = Math.floor(s / 16 ** 5);
  if (cat === 8 && (s % 16 ** 5) >> 16 === 14) return 'Kráľovská postupka';
  return HAND_NAMES[cat];
}

// A fast random source so results are repeatable for the same cards.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Share of the pot you win on average (ties split), between 0 and 1.
export function equity(hole, board, opponents, trials = 2500) {
  const known = new Set([...hole, ...board]);
  const deck = fullDeck().filter((c) => !known.has(c)).map(parse);
  const me = hole.map(parse), shown = board.map(parse);
  let seed = 7;
  for (const ch of [...hole, ...board].join()) seed = (seed * 31 + ch.charCodeAt(0)) | 0;
  const rand = rng(seed + opponents);
  const need = 5 - shown.length + opponents * 2;
  let won = 0;

  for (let t = 0; t < trials; t++) {
    // Partial shuffle: only the first `need` cards are drawn.
    for (let i = 0; i < need; i++) {
      const j = i + Math.floor(rand() * (deck.length - i));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    const fullBoard = shown.concat(deck.slice(0, 5 - shown.length));
    const mine = score(me.concat(fullBoard));
    let ties = 0, lost = false;
    for (let o = 0; o < opponents; o++) {
      const k = 5 - shown.length + o * 2;
      const theirs = score([deck[k], deck[k + 1], ...fullBoard]);
      if (theirs > mine) { lost = true; break; }
      if (theirs === mine) ties++;
    }
    if (!lost) won += 1 / (ties + 1);
  }
  return won / trials;
}

const STAGE = { 0: 'Pred flopom', 3: 'Flop', 4: 'Turn', 5: 'River' };

export function advise({ mine, table }, { opponents = 3 } = {}) {
  if (mine.length < 2) {
    return {
      action: 'Ukáž karty',
      tone: 'wait',
      lines: [mine.length ? `Vidím len ${showAll(mine)}. Potrebujem obe tvoje karty.` : 'Namier kameru na svoje dve karty.'],
    };
  }
  if (mine.length > 2) {
    return {
      action: 'Priveľa kariet',
      tone: 'wait',
      lines: [`Ako tvoje vidím ${mine.length} karty: ${showAll(mine)}.`, 'Ťukni na karty na stole, aby sa označili ako stôl.'],
    };
  }
  if (table.length > 5 || table.length === 1 || table.length === 2) {
    const why = table.length > 5 ? 'Na stole môže byť najviac 5 kariet.' : 'Na stole bývajú 3, 4 alebo 5 kariet.';
    return { action: 'Skontroluj stôl', tone: 'wait', lines: [`Stôl: ${showAll(table)}`, why] };
  }

  const e = equity(mine, table, opponents);
  const fair = 1 / (opponents + 1);
  const pct = Math.round(e * 100);
  const lines = [
    `${STAGE[table.length]} · Ty: ${showAll(mine)}${table.length ? ` · Stôl: ${showAll(table)}` : ''}`,
    `${table.length ? handName([...mine, ...table]) : handName(mine)} · vyhráš asi ${pct} % (proti ${opponents} ${opponents === 1 ? 'hráčovi' : 'hráčom'})`,
  ];

  let action, tone;
  if (e >= Math.max(fair * 1.6, 0.5)) { action = 'Zvýš'; tone = 'go'; }
  else if (e >= fair) { action = 'Dorovnaj'; tone = 'go'; }
  else if (e >= fair * 0.7) { action = 'Check alebo lacno dorovnaj'; tone = 'wait'; }
  else { action = 'Zahoď'; tone = 'stop'; }

  // Calling pays off when your chance to win beats the share of the pot you put in.
  if (action !== 'Zvýš' && e < 0.95) {
    const maxCall = Math.round((e / (1 - e)) * 100);
    lines.push(`Dorovnať sa oplatí, ak stávka je najviac ${maxCall} % banku.`);
  }
  if (action === 'Zahoď') lines.push('Ak nikto nestavil, môžeš dať check zadarmo.');
  return { action, tone, lines, equity: e };
}
