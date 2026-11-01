// Run with: npm test   (or: node --test)
import test from 'node:test';
import assert from 'node:assert/strict';
import * as bj from '../games/blackjack.js';
import * as poker from '../games/poker.js';
import * as zolik from '../games/zolik.js';

test('blackjack totals count aces as 1 or 11', () => {
  assert.deepEqual(bj.total(['AS', '6H']), { sum: 17, soft: true });
  assert.deepEqual(bj.total(['AS', '6H', '10C']), { sum: 17, soft: false });
  assert.deepEqual(bj.total(['AS', 'AD', '9C']), { sum: 21, soft: true });
  assert.deepEqual(bj.total(['KS', 'QH', '5C']), { sum: 25, soft: false });
});

test('blackjack basic strategy spot checks', () => {
  const d = (cards, up, o) => bj.decide(cards, up, o);
  assert.equal(d(['10S', '6H'], 10), 'H');   // 16 vs 10: hit
  assert.equal(d(['10S', '6H'], 6), 'S');    // 16 vs 6: stand
  assert.equal(d(['10S', '2H'], 3), 'H');    // 12 vs 3: hit
  assert.equal(d(['10S', '2H'], 4), 'S');    // 12 vs 4: stand
  assert.equal(d(['6S', '5H'], 10), 'D');    // 11 vs 10: double
  assert.equal(d(['6S', '5H'], 11), 'H');    // 11 vs A, dealer stands soft 17
  assert.equal(d(['6S', '5H'], 11, { hitSoft17: true }), 'D');
  assert.equal(d(['AS', '7H'], 9), 'H');     // soft 18 vs 9: hit
  assert.equal(d(['AS', '7H'], 7), 'S');
  assert.equal(d(['AS', '7H'], 5), 'Ds');
  assert.equal(d(['8S', '8H'], 10), 'P');    // always split eights
  assert.equal(d(['AS', 'AH'], 11), 'P');    // and aces
  assert.equal(d(['KS', 'QH'], 6), 'S');     // never split tens
  assert.equal(d(['9S', '9H'], 7), 'S');
  assert.equal(d(['9S', '9H'], 8), 'P');
  assert.equal(d(['5S', '5H'], 9), 'D');     // fives play as 10
});

test('blackjack advice wording', () => {
  assert.equal(bj.advise({ mine: ['10S', '6H'], table: ['KD'] }).action, 'Ber kartu');
  assert.equal(bj.advise({ mine: ['10S', '6H'], table: ['5D'] }).action, 'Stoj');
  assert.equal(bj.advise({ mine: ['AS', 'KH'], table: ['5D'] }).action, 'Blackjack!');
  assert.equal(bj.advise({ mine: ['10S', '6H', '9C'], table: ['5D'] }).action, 'Prehral si');
  // Double is not allowed after the first two cards.
  assert.equal(bj.advise({ mine: ['3S', '4H', '4C'], table: ['6D'] }).action, 'Ber kartu');
  assert.equal(bj.advise({ mine: ['10S', '6H'], table: [] }).action, 'Krupiér?');
});

test('poker hand names', () => {
  assert.equal(poker.handName(['AS', 'KS', 'QS', 'JS', '10S', '2H', '3D']), 'Kráľovská postupka');
  assert.equal(poker.handName(['9S', 'KS', 'QS', 'JS', '10S', '2H', '3D']), 'Postupka vo farbe');
  assert.equal(poker.handName(['AS', '2H', '3D', '4C', '5S', 'KD', 'QD']), 'Postupka');
  assert.equal(poker.handName(['AS', 'AH', 'AD', 'KC', 'KS', '2D', '3D']), 'Full house');
  assert.equal(poker.handName(['AS', 'AH', 'KD', 'KC', 'QS', 'QD', '3D']), 'Dva páry');
  assert.equal(poker.handName(['2H', '7H', '9H', 'JH', 'KH', 'AS', 'AD']), 'Farba');
  assert.equal(poker.handName(['7S', '7H', '7D', '7C', 'KH', 'AS', 'AD']), 'Štvorica');
  assert.equal(poker.handName(['AS', 'AH']), 'Pár');
});

test('poker scores order hands correctly', () => {
  const s = (codes) => poker.score(codes.map((c) => ({ rank: { J: 11, Q: 12, K: 13, A: 14 }[c.slice(0, -1)] ?? Number(c.slice(0, -1)), suit: c.at(-1) })));
  assert.ok(s(['AS', 'AH', '2D', '3C', '7S']) > s(['KS', 'KH', 'QD', 'JC', '9S']));  // pair of aces beats kings
  assert.ok(s(['2S', '3H', '4D', '5C', '6S']) > s(['AS', '2H', '3D', '4C', '5S']));  // 6-high straight beats the wheel
  assert.ok(s(['AS', 'AH', 'KD', '3C', '2S']) > s(['AS', 'AH', 'QD', 'JC', '9S']));  // kicker
});

test('poker equity is about right for known spots', () => {
  const aa = poker.equity(['AS', 'AH'], [], 1, 4000);
  assert.ok(aa > 0.8 && aa < 0.89, `AA heads-up ${aa}`);           // about 85 %
  const low = poker.equity(['7C', '2D'], [], 1, 4000);
  assert.ok(low > 0.28 && low < 0.40, `72o heads-up ${low}`);      // about 35 %
  const nuts = poker.equity(['AS', 'KS'], ['QS', 'JS', '10S'], 3, 500);
  assert.equal(nuts, 1);
  assert.equal(poker.advise({ mine: ['AS', 'AH'], table: [] }, { opponents: 1 }).action, 'Zvýš');
  assert.equal(poker.advise({ mine: ['7C', '2D'], table: [] }, { opponents: 5 }).action, 'Zahoď');
});

test('zolik finds runs and sets', () => {
  const r = zolik.bestMelds(['7H', '8H', '9H', 'KS', 'KD', 'KC', '2C'], 'cards');
  assert.equal(r.melds.length, 2);
  assert.deepEqual(r.left, ['2C']);
  assert.equal(r.points, 7 + 8 + 9 + 30);
});

test('zolik uses jokers to fill gaps and counts their value', () => {
  const r = zolik.bestMelds(['5S', '7S', 'JK'], 'cards');
  assert.equal(r.melds.length, 1);
  assert.deepEqual(r.melds[0].shown, ['5S', 'JK', '7S']);
  assert.equal(r.points, 18);
});

test('zolik ace is low or high but does not wrap', () => {
  assert.equal(zolik.bestMelds(['AH', '2H', '3H'], 'points').points, 6);
  assert.equal(zolik.bestMelds(['QH', 'KH', 'AH'], 'points').points, 30);
  assert.equal(zolik.bestMelds(['KH', 'AH', '2H'], 'cards').melds.length, 0);
});

test('zolik sets need different suits, two decks can share a card', () => {
  assert.equal(zolik.bestMelds(['7H', '7H', '7S'], 'cards').melds.length, 0);
  // Two 7♥ from two decks: one in a run, one in a set.
  const r = zolik.bestMelds(['5H', '6H', '7H', '7H', '7S', '7D'], 'cards');
  assert.equal(r.cards, 6);
});

test('zolik opening needs 42 points', () => {
  const low = zolik.advise({ mine: ['2H', '3H', '4H', '5S', '5D', '5C', 'KD', '9C'] }, { openPoints: 42 });
  assert.match(low.action, /^Zahoď/);
  const high = zolik.advise({ mine: ['10H', 'JH', 'QH', 'KS', 'KD', 'KC', '2D', '9C'] }, { openPoints: 42 });
  assert.equal(high.action, 'Vylož (60 b.)');
  assert.ok(['2D', '9C'].includes(high.discard));
});

test('zolik discards the loneliest high card and lists useful draws', () => {
  // 8♣ sits next to 9♣; K♦ has no partner, so it goes.
  const a = zolik.advise({ mine: ['7H', '8H', '9H', '8C', '9C', 'KD'] }, { opened: true });
  assert.equal(a.discard, 'KD');
  assert.ok(zolik.wanted(['7H', '8H', '9H', '8C', '9C', 'KD']).includes('10C'));
});

test('zolik finishes when one card is left', () => {
  const a = zolik.advise({ mine: ['7H', '8H', '9H', 'KS', 'KD', 'KC', '2C'] }, { opened: true });
  assert.equal(a.action, 'Končíš! Zahoď 2♣');
});

test('zolik search stays fast on a full two-deck hand with jokers', () => {
  const hand = ['3H', '4H', '6H', '7H', '7S', '7D', '9C', '10C', 'QC', 'KC', 'AD', '2D', 'JK', 'JK'];
  const t = performance.now();
  zolik.advise({ mine: hand }, {});
  const ms = performance.now() - t;
  assert.ok(ms < 1500, `took ${ms} ms`);
});
