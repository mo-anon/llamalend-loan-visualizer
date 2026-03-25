# llamalend-loan-visualizer

Generate time-series chart videos of [Curve LLAMMA](https://docs.curve.fi/lending/overview/) soft-liquidation loans. Each frame shows the loan's band composition (collateral vs crvUSD), oracle price, health factor, and key events (borrows, repayments, liquidations) — stitched together into an MP4 video.

## How it works

1. Fetches on-chain loan state at each sampled block via RPC multicall
2. Renders a chart frame per block with matplotlib
3. Encodes frames into an MP4 with ffmpeg

Loan events (borrow, repay, liquidation, etc.) are auto-fetched from the [Curve Prices API](https://prices.curve.fi) and marked on the chart.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [ffmpeg](https://ffmpeg.org/) installed and on PATH
- An Ethereum RPC endpoint (e.g. [Alchemy](https://www.alchemy.com/))

## Setup

```bash
git clone https://github.com/<your-username>/llamalend-loan-visualizer.git
cd llamalend-loan-visualizer
```

Create a `.env` file with your RPC endpoint(s) and optionally an Etherscan API key (needed if you specify block ranges as datetime strings):

```
RPC_URL_1=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
RPC_URL_2=https://eth-mainnet.g.alchemy.com/v2/BACKUP_KEY
ETHERSCAN_API_KEY=YOUR_KEY
```

Multiple `RPC_URL_*` entries enable automatic round-robin failover on rate limits.

## Usage

### Web UI

```bash
uv run python app.py
```

Open http://localhost:8000, fill in the controller and user addresses, configure the block range, and click Generate. Logs stream in real time and the video plays on completion.

### CLI

Edit the constants at the top of `visualize_loan.py` (addresses, block range, step size), then:

```bash
uv run python visualize_loan.py
```

### Block range options

The block range can be specified three ways:

| START_BLOCK | END_BLOCK | DURATION | Result |
|---|---|---|---|
| 24378003 | 24378883 | — | Explicit range |
| — | 24378883 | 880 | End block, go backwards |
| 24378883 | — | 100 | Start block, go forwards |

`START_BLOCK` and `END_BLOCK` accept block numbers (int), UTC datetime strings (e.g. `"2026-01-01 12:00"`), or `None` for latest.

`BLOCK_STEP` controls sampling density — lower values produce smoother videos with more frames.

## Output

Each run produces a directory under `output/`:

```
output/{date}_{collateral}-{base}_{user_short}/
├── video.mp4          # Final video (H.264, 50fps)
└── block_data.json    # Cached RPC data (re-render without re-fetching)
```
