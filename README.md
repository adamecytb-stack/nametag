# Karty

A phone camera that reads playing cards and tells you what to play in **blackjack**, **Texas Hold'em poker** and **žolík**. It draws a label on every card it recognises and shows the advice at the bottom of the screen. The app is in Slovak.

Open **https://adamecytb-stack.github.io/nametag/** in Safari on an iPhone, tap Share, then **Add to Home Screen**.

## How to use it

1. Pick the game at the top: Blackjack, Poker or Žolík.
2. Point the back camera at the cards so their **corners** (the small number and suit) are visible. A fanned hand works, because only the corner is needed.
3. Every card it is sure about gets a label like `Q♥ 92 %`. Yellow = your card.
4. **Tap a card** to say whose it is. Each tap moves it to the next owner:
   - Blackjack: yours → dealer (`krupiér`, blue) → ignored
   - Poker: yours → table (`stôl`, blue) → ignored
   - Žolík: yours → ignored
5. Read the advice in the panel at the bottom.
6. **Nové kolo** (new round) forgets the owners and any cards you added by hand.
7. **+ Karta** adds a card by hand, for a card the camera cannot see or read correctly. In žolík this is also how you add jokers. Tap the card's chip in the panel to remove it.
8. The middle button freezes the picture, so you can read the advice calmly.

### What the advice is based on

| Game | Advice |
| --- | --- |
| Blackjack | Basic strategy for 4–8 decks, doubling after a split allowed, no surrender. Settings has a switch for dealers who hit a soft 17. Shows your total, the dealer's up card and *Ber kartu / Stoj / Zdvojnásob / Rozdeľ*. |
| Poker | Texas Hold'em. Your best hand name and your chance to win, estimated by dealing the unknown cards 2,500 times against the number of opponents set in Settings. Says *Zvýš / Dorovnaj / Check / Zahoď* and up to what bet size a call pays off. |
| Žolík | 2 decks + 4 jokers. Finds the best runs (3+ cards in a row of one suit; A-2-3 or Q-K-A, no K-A-2) and sets (3–4 of a rank in different suits). J/Q/K/A = 10 points, A in A-2-3 = 1. Checks the 42-point opening (changeable), suggests a discard and lists cards that would help you. Every meld keeps at least 2 real cards. |

## Limits

- The model knows the 52 standard cards with **A K Q J** indices. Decks that print other letters (for example a French deck with 1 / R / D / V) are read badly or not at all.
- It cannot see **jokers**. Add them with + Karta.
- In žolík (two decks), two identical cards lying very close together count as one. Add the second one with + Karta.
- Faded, glossy or very small cards give weaker results. The *Rozpoznávanie* slider in Settings trades missed cards for wrong cards.
- It is a learning aid. Using a device for advice at a casino table is not allowed, and in some places it is illegal.

## Where things run

Everything runs on the phone. No picture leaves it.

| Job | What |
| --- | --- |
| Reading cards | YOLOv8n card-corner detector (`models/cards.onnx`, 12 MB, MIT license, see `models/README.md`), run by [ONNX Runtime Web](https://onnxruntime.ai/). It uses the GPU through WebGPU where Safari supports it, otherwise the CPU. |
| Advice | Plain JavaScript in `games/` |

The model ships with the app. ONNX Runtime is downloaded from jsDelivr on the first launch, then everything is cached for offline use.

## Files

```
index.html, styles.css   the screen
app.js                   camera, following cards between frames, drawing, settings
detector.js              runs the model and groups card corners into cards
cards.js                 card codes ('QH', '10S', 'JK' = joker) and how they are shown
games/blackjack.js       basic strategy
games/poker.js           hand ranking and win chance
games/zolik.js           meld search, opening, discard
tests/games.test.mjs     tests for the three games
tools/export_model.py    turns a YOLO .pt model into models/cards.onnx (Python)
tools/try_model.py       runs models/cards.onnx on photos from your computer (Python)
```

## Run it locally

```bash
python -m http.server 8791
```

Then open http://localhost:8791. The camera needs https or localhost.

Tests for the game logic need Node.js 20 or newer:

```bash
npm test
```

To check the model on your own photos (Python):

```bash
pip install onnxruntime pillow numpy
python tools/try_model.py my_cards.jpg --save checked/
```
