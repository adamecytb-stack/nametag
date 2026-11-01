// Blackjack basic strategy for a shoe of 4–8 decks, double after split allowed,
// no surrender. `hitSoft17` switches between the two common dealer rules.
import { parse, showAll } from '../cards.js';

const HIT = 'Ber kartu', STAND = 'Stoj', DOUBLE = 'Zdvojnásob', SPLIT = 'Rozdeľ';

// Value of one card for blackjack: J/Q/K are 10, an ace counts as 11 here.
const value = (code) => {
  const { rank } = parse(code);
  if (rank === 14) return 11;
  return Math.min(rank, 10);
};

export function total(codes) {
  let sum = 0, aces = 0;
  for (const c of codes) {
    const v = value(c);
    sum += v;
    if (v === 11) aces++;
  }
  while (sum > 21 && aces) { sum -= 10; aces--; }
  return { sum, soft: aces > 0 };
}

// d = dealer up card value, 2..11 (11 = ace). Returns 'H', 'S', 'D' (double, else hit),
// 'Ds' (double, else stand) or 'P' (split).
export function decide(cards, d, { hitSoft17 = false } = {}) {
  const { sum, soft } = total(cards);
  const two = cards.length === 2;
  const v0 = two ? value(cards[0]) : 0;

  if (two && v0 === value(cards[1])) {
    const split = {
      11: true,
      10: false,
      9: d <= 9 && d !== 7,
      8: true,
      7: d <= 7,
      6: d <= 6,
      5: false,
      4: d === 5 || d === 6,
      3: d <= 7,
      2: d <= 7,
    }[v0];
    if (split) return 'P';
  }

  if (soft) {
    if (sum >= 20) return 'S';
    if (sum === 19) return hitSoft17 && d === 6 ? 'Ds' : 'S';
    if (sum === 18) {
      if (d <= 6) return hitSoft17 || d >= 3 ? 'Ds' : 'S';
      return d <= 8 ? 'S' : 'H';
    }
    if (sum === 17) return d >= 3 && d <= 6 ? 'D' : 'H';
    if (sum >= 15) return d >= 4 && d <= 6 ? 'D' : 'H';
    return d >= 5 && d <= 6 ? 'D' : 'H';
  }

  if (sum >= 17) return 'S';
  if (sum >= 13) return d <= 6 ? 'S' : 'H';
  if (sum === 12) return d >= 4 && d <= 6 ? 'S' : 'H';
  if (sum === 11) return d <= 10 || hitSoft17 ? 'D' : 'H';
  if (sum === 10) return d <= 9 ? 'D' : 'H';
  if (sum === 9) return d >= 3 && d <= 6 ? 'D' : 'H';
  return 'H';
}

const WHY = {
  P: 'Pár sa oplatí rozdeliť na dve ruky.',
  D: 'Dobrá pozícia: zdvojnásob stávku a zober presne jednu kartu.',
  Ds: 'Dobrá pozícia: zdvojnásob stávku a zober presne jednu kartu.',
};

function describe({ sum, soft }) {
  return soft && sum < 21 ? `${sum} (mäkkých, eso = 11)` : String(sum);
}

export function advise({ mine, table }, opts = {}) {
  if (!mine.length) {
    return { action: 'Ukáž karty', tone: 'wait', lines: ['Namier kameru na svoje karty.'] };
  }
  const me = total(mine);
  const lines = [`Ty: ${showAll(mine)} = ${describe(me)}`];

  if (me.sum > 21) return { action: 'Prehral si', tone: 'stop', lines: [...lines, 'Máš viac ako 21.'] };
  if (mine.length === 2 && me.sum === 21) return { action: 'Blackjack!', tone: 'go', lines: [...lines, 'Máš 21 z dvoch kariet. Stoj.'] };

  if (!table.length) {
    lines.push('Ťukni na kartu krupiéra, aby som vedel, čo má.');
    return { action: me.sum >= 17 ? STAND : 'Krupiér?', tone: me.sum >= 17 ? 'stop' : 'wait', lines };
  }

  const up = table[0];
  const d = value(up);
  if (table.length > 1) {
    const dealer = total(table);
    lines.push(`Krupiér: ${showAll(table)} = ${describe(dealer)}`);
  } else {
    lines.push(`Krupiér ukazuje: ${showAll([up])}`);
  }

  const move = decide(mine, d, opts);
  const canDouble = mine.length === 2;
  let action;
  if (move === 'P') action = SPLIT;
  else if (move === 'D') action = canDouble ? DOUBLE : HIT;
  else if (move === 'Ds') action = canDouble ? DOUBLE : STAND;
  else action = move === 'H' ? HIT : STAND;

  if (action === HIT) lines.push(d >= 7 ? 'Krupiér má silnú kartu, s týmto súčtom potrebuješ viac.' : 'Ďalšia karta ťa nemôže pokaziť, alebo je to len malé riziko.');
  else if (action === STAND) lines.push(d <= 6 && me.sum < 17 ? 'Krupiér má slabú kartu a pravdepodobne prekročí 21. Neriskuj.' : 'Tvoj súčet je dosť dobrý.');
  else lines.push(WHY[move]);

  if (table.length === 1 && d === 11 && mine.length === 2) lines.push('Poistenie (insurance) neber, dlhodobo sa neoplatí.');

  const tone = action === STAND ? 'stop' : 'go';
  return { action, tone, lines };
}
