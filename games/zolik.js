// Žolík (Czech/Slovak rummy), common rules:
//   2 decks + 4 jokers, 12 cards each (13 for the player who starts).
//   Postupka (run): 3+ cards of one suit in a row. The ace is low (A-2-3) or
//   high (Q-K-A), never both (no K-A-2).
//   Skupina (set): 3 or 4 cards of one rank, every suit different.
//   2–10 count their number, J/Q/K count 10, A counts 10 (1 in A-2-3),
//   a joker counts as the card it stands for.
//   The first lay-down needs at least 42 points.
// This app also keeps at least 2 real cards in every meld, the stricter house rule.
import { parse, codeOf, show, fullDeck, JOKER, SUITS } from '../cards.js';

const MIN_NATURAL = 2;

const setValue = (rank) => (rank >= 11 ? 10 : rank);
// Position in a run: 1 = ace low, 2..10, 11 = J, 12 = Q, 13 = K, 14 = ace high.
const runValue = (pos) => (pos === 1 ? 1 : pos >= 11 ? 10 : pos);
const runCode = (pos, suit) => codeOf(pos === 1 ? 14 : pos, suit);

function* subsets(items, size, start = 0, picked = []) {
  if (picked.length === size) { yield picked; return; }
  for (let i = start; i <= items.length - (size - picked.length); i++) {
    yield* subsets(items, size, i + 1, [...picked, items[i]]);
  }
}

// Every meld that can be built from the cards in `count` (code -> how many)
// with at most `jokers` jokers.
export function possibleMelds(count, jokers) {
  const melds = [];

  for (let rank = 2; rank <= 14; rank++) {
    const suits = SUITS.filter((s) => count[codeOf(rank, s)]);
    for (let k = MIN_NATURAL; k <= suits.length; k++) {
      for (const chosen of subsets(suits, k)) {
        for (let j = 0; j <= jokers && k + j <= 4; j++) {
          if (k + j < 3) continue;
          const cards = chosen.map((s) => codeOf(rank, s));
          melds.push({
            kind: 'set',
            cards,
            jokers: j,
            points: setValue(rank) * (k + j),
            shown: [...cards, ...Array(j).fill(JOKER)],
          });
        }
      }
    }
  }

  for (const suit of SUITS) {
    for (let start = 1; start <= 12; start++) {
      for (let end = start + 2; end <= 14; end++) {
        if (start === 1 && end === 14) continue;
        const positions = [];
        for (let p = start; p <= end; p++) positions.push(p);
        const present = positions.filter((p) => count[runCode(p, suit)]);
        const gaps = positions.length - present.length;
        if (gaps > jokers) break; // longer runs from this start only need more jokers
        if (present.length < MIN_NATURAL) continue;
        // Usually every card we hold goes in, but a joker can stand in for one
        // we hold, so that card stays free for another meld.
        const spare = jokers - gaps;
        for (let swap = 0; swap <= Math.min(spare, 1); swap++) {
          for (const swapped of subsets(present, swap)) {
            const naturals = present.filter((p) => !swapped.includes(p));
            if (naturals.length < MIN_NATURAL) continue;
            melds.push({
              kind: 'run',
              cards: naturals.map((p) => runCode(p, suit)),
              jokers: gaps + swap,
              points: positions.reduce((sum, p) => sum + runValue(p), 0),
              shown: positions.map((p) => (naturals.includes(p) ? runCode(p, suit) : JOKER)),
            });
          }
        }
      }
    }
  }
  return melds;
}

// Chooses melds that do not share cards. `goal` 'points' maximises points
// (to reach the opening), 'cards' maximises how many cards get laid down.
export function bestMelds(hand, goal = 'cards') {
  const jokers = hand.filter((c) => c === JOKER).length;
  const naturals = hand.filter((c) => c !== JOKER);
  const codes = [...new Set(naturals)].sort();
  const idx = new Map(codes.map((c, i) => [c, i]));
  const start = codes.map((c) => naturals.filter((x) => x === c).length);
  const count = Object.fromEntries(codes.map((c, i) => [c, start[i]]));

  const melds = possibleMelds(count, jokers).map((m) => ({ ...m, need: m.cards.map((c) => idx.get(c)) }));
  const withCard = codes.map((_, i) => melds.filter((m) => m.need.includes(i)));
  const worth = (cards, points) => (goal === 'cards' ? cards * 1000 + points : points * 1000 + cards);

  const memo = new Map();
  function solve(counts, jk) {
    const key = counts.join() + '|' + jk;
    if (memo.has(key)) return memo.get(key);
    const first = counts.findIndex((n) => n > 0);
    let best = { score: 0, cards: 0, points: 0, melds: [] };
    if (first >= 0) {
      counts[first]--;
      best = solve(counts, jk); // this card stays in hand
      counts[first]++;
      for (const m of withCard[first]) {
        if (m.jokers > jk || m.need.some((i) => counts[i] < 1)) continue;
        for (const i of m.need) counts[i]--;
        const rest = solve(counts, jk - m.jokers);
        for (const i of m.need) counts[i]++;
        const cards = rest.cards + m.cards.length + m.jokers, points = rest.points + m.points;
        const score = worth(cards, points);
        if (score > best.score) best = { score, cards, points, melds: [m, ...rest.melds] };
      }
    }
    memo.set(key, best);
    return best;
  }

  const best = solve(start, jokers);
  // Cards not used by any meld.
  const left = [...naturals];
  let jokersLeft = jokers;
  for (const m of best.melds) {
    for (const c of m.cards) left.splice(left.indexOf(c), 1);
    jokersLeft -= m.jokers;
  }
  return { melds: best.melds, points: best.points, cards: best.cards, left, jokersLeft };
}

function partners(card, others) {
  const a = parse(card);
  let n = 0;
  for (const o of others) {
    if (o === card) continue;
    const b = parse(o);
    if (b.joker) { n++; continue; }
    if (a.rank === b.rank && a.suit !== b.suit) n++;
    else if (a.suit === b.suit) {
      const d = Math.abs(a.rank - b.rank);
      // The ace also sits next to the 2 and 3.
      const low = a.rank === 14 || b.rank === 14 ? Math.abs((a.rank % 13) - (b.rank % 13)) : 99;
      if ((d > 0 && d <= 2) || (low > 0 && low <= 2)) n++;
    }
  }
  return n;
}

// The leftover card least likely to become part of a meld; high cards go first
// because they cost the most if you are caught holding them.
export function pickDiscard(left, jokersLeft = 0) {
  if (!left.length) return null;
  const pool = [...left, ...Array(jokersLeft).fill(JOKER)];
  const keep = (c) => partners(c, pool) * 15 - setValue(parse(c).rank);
  return [...left].sort((a, b) => keep(a) - keep(b) || setValue(parse(b).rank) - setValue(parse(a).rank))[0];
}

// Cards that would leave fewer unused cards in hand if you drew them.
export function wanted(hand) {
  const base = bestMelds(hand, 'cards');
  const before = base.left.length;
  const out = [];
  for (const c of fullDeck()) {
    const after = bestMelds([...hand, c], 'cards').left.length;
    if (after < before) out.push({ code: c, gain: before - after });
  }
  return out.sort((a, b) => b.gain - a.gain).map((w) => w.code);
}

const showMeld = (m) => m.shown.map(show).join(' ');

export function advise({ mine }, { openPoints = 42, opened = false } = {}) {
  if (!mine.length) {
    return { action: 'Ukáž karty', tone: 'wait', lines: ['Rozlož karty do vejára tak, aby boli vidieť ich rohy.'] };
  }
  const structure = bestMelds(mine, 'cards');
  const lines = [];
  const jokerCount = mine.filter((c) => c === JOKER).length;
  lines.push(`Vidím ${mine.length} ${mine.length < 5 ? 'karty' : 'kariet'}${jokerCount ? `, z toho ${jokerCount} žolík${jokerCount > 1 ? 'y' : ''}` : ''}.`);

  if (!opened) {
    const open = bestMelds(mine, 'points');
    if (open.points >= openPoints) {
      const discard = pickDiscard(open.left, open.jokersLeft);
      lines.push(...open.melds.map(showMeld));
      if (discard) lines.push(`Potom zahoď ${show(discard)}.`);
      return { action: `Vylož (${open.points} b.)`, tone: 'go', lines, melds: open.melds, discard };
    }
    const discard = pickDiscard(structure.left, structure.jokersLeft);
    lines.push(`V kombináciách máš ${open.points} b., na vyloženie treba ${openPoints}.`);
    if (structure.melds.length) lines.push(`Drž: ${structure.melds.map(showMeld).join('  ·  ')}`);
    const want = wanted(mine).slice(0, 6);
    if (want.length) lines.push(`Pomohli by ti: ${want.map(show).join(' ')}`);
    return {
      action: discard ? `Zahoď ${show(discard)}` : 'Ťahaj',
      tone: discard ? 'stop' : 'wait',
      lines,
      melds: structure.melds,
      discard,
    };
  }

  const discard = pickDiscard(structure.left, structure.jokersLeft);
  if (structure.melds.length) lines.push(...structure.melds.map(showMeld));
  const unused = structure.left.length + structure.jokersLeft;
  if (unused === 0) return { action: 'Vylož všetko!', tone: 'go', lines: [...lines, 'Všetky karty sú v kombináciách.'], melds: structure.melds };
  if (unused === 1 && discard) return { action: `Končíš! Zahoď ${show(discard)}`, tone: 'go', lines, melds: structure.melds, discard };
  if (structure.melds.length) lines.push(`Ostáva ti ${unused} ${unused < 5 ? 'karty' : 'kariet'}.`);
  const want = wanted(mine).slice(0, 6);
  if (want.length) lines.push(`Pomohli by ti: ${want.map(show).join(' ')}`);
  return {
    action: discard ? (structure.melds.length ? `Vylož a zahoď ${show(discard)}` : `Zahoď ${show(discard)}`) : 'Ťahaj',
    tone: 'go',
    lines,
    melds: structure.melds,
    discard,
  };
}
