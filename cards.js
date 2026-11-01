// Cards are short codes: rank then suit, the same labels the model uses.
//   'AS' = ace of spades, '10H' = ten of hearts, 'QD' = queen of diamonds.
// 'JK' is a joker (žolík). The model cannot see jokers; they are added by hand.

export const SUITS = ['S', 'H', 'D', 'C'];
export const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
export const JOKER = 'JK';

const SYMBOL = { S: '♠', H: '♥', D: '♦', C: '♣' };
const RANK_VALUE = { J: 11, Q: 12, K: 13, A: 14 };

export function parse(code) {
  if (code === JOKER) return { code, joker: true, rank: 0, suit: null };
  const suit = code.at(-1);
  const r = code.slice(0, -1);
  return { code, joker: false, rank: RANK_VALUE[r] ?? Number(r), suit };
}

export function codeOf(rank, suit) {
  const r = { 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 1: 'A' }[rank] ?? String(rank);
  return r + suit;
}

export const isRed = (code) => code.endsWith('H') || code.endsWith('D');

export function show(code) {
  if (code === JOKER) return 'Žolík';
  return code.slice(0, -1) + SYMBOL[code.at(-1)];
}

export const showAll = (codes) => codes.map(show).join(' ');

export const fullDeck = () => SUITS.flatMap((s) => RANKS.map((r) => r + s));
