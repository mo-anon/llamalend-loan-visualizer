"""
Visualize a Curve LLAMMA soft-liquidation loan as a time-series chart video.

Pipeline:
  1. Resolve block range (supports block numbers, datetime strings, or "latest")
  2. FIRST PASS  — Fetch on-chain data for each block via RPC + multicall
  3. SECOND PASS — Render one PNG frame per block (incremental; figure persists)
  4. Encode frames into an MP4 with ffmpeg, then delete the frames

Output goes to:  output/{date}_{market}_{user_short}/video.mp4
Cached RPC data: output/{date}_{market}_{user_short}/block_data.json

Usage:
  uv run python visualize_loan.py
"""

import concurrent.futures
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Union

import numpy as np

import matplotlib
matplotlib.use('Agg')  # non-GUI backend — no window, faster saves
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager, rcParams
from web3 import Web3
from web3.exceptions import ContractLogicError, BlockNotFound

from llamma_loan import (
    RPCManager,
    CurveLoanGetter,
    PRECISION,
    resolve_block,
    fetch_events,
)

BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# CONFIGURATION — edit these to visualize a different loan
# =============================================================================

# Curve lending controller address — the AMM (LLAMMA) address is derived
# automatically by calling controller.amm() on-chain.
CONTROLLER_ADDRESS = "0xEdA215b7666936DEd834f76f3fBC6F323295110A"

# Borrower address to track
USER_ADDRESS = "0x9F2F1E3dedEeA341c479d3bD5D5775De4448d8Bb"

# Where to write output (gitignored)
OUTPUT_BASE = BASE_DIR / "output"

# Block range — three ways to specify:
#
#   1) Explicit start + end:
#        START_BLOCK = 24378003
#        END_BLOCK   = 24378883
#        DURATION    = None
#
#   2) End + go backwards by DURATION blocks:
#        START_BLOCK = None
#        END_BLOCK   = 24378883   (or None for latest block)
#        DURATION    = 880
#
#   3) Start + go forwards by DURATION blocks:
#        START_BLOCK = 24378883
#        END_BLOCK   = None
#        DURATION    = 100
#
# Each of START_BLOCK / END_BLOCK can be:
#   - int    → block number (e.g. 24378003)
#   - str    → UTC datetime, resolved via Etherscan API (e.g. "2026-01-01 12:00")
#   - None   → latest block (END_BLOCK) / derive from DURATION (START_BLOCK)
#
# Datetime strings require ETHERSCAN_API_KEY in .env (free at etherscan.io).
START_BLOCK =24343202
END_BLOCK = 24741632
DURATION = None
BLOCK_STEP = 500  # sample every Nth block — lower = more frames, smoother video
MAX_WORKERS = 6   # parallel RPC fetchers — set to 1 to disable parallelism


# Events to mark on the chart — vertical lines + info panel.
# Set AUTO_FETCH_EVENTS=True to auto-fetch from the Curve Prices API.
# If EVENTS is non-empty it is used as-is regardless of AUTO_FETCH_EVENTS.
AUTO_FETCH_EVENTS = True
EVENTS = []


# =============================================================================
# Chart styling (colors, fonts, event labels) — rarely needs changing
# =============================================================================

EVENT_LABELS = {
    "repay_debt": "Repayment",
    "borrow": "Borrow",
    "add_collateral": "Add Collateral",
    "remove_collateral": "Remove Collateral",
    "closed": "Loan Closed",
    "liquidated": "Liquidated",
}
EVENT_ICONS = {
    "repay_debt": "↩",
    "add_collateral": "➕",
    "remove_collateral": "➖",
    "closed": "●",
    "liquidated": "✖",
}
EVENT_STYLE = {
    "repay_debt": {"color": "#00FF00", "linestyle": ":", "linewidth": 1.0},
    "borrow": {"color": "#1E90FF", "linestyle": ":", "linewidth": 1.0},
    "borrow_debt": {"color": "#1E90FF", "linestyle": ":", "linewidth": 1.0},
}

CHART_COLORS = {
    "crvusd": {"fill": "#A8EFC6", "edge": "#67D696"},
    "collateral": {"fill": "#ACBEF1", "edge": "#86A2F0"},
    "health": {"fill": "#FFD88B", "edge": "#FFC300"},
}

font_path = BASE_DIR / "MonaSans-Regular.ttf"
font_manager.fontManager.addfont(str(font_path))
monasans_font = font_manager.FontProperties(fname=str(font_path))
rcParams['font.sans-serif'] = [monasans_font.get_name()]


# =============================================================================
# Pipeline config
# =============================================================================

@dataclass
class PipelineConfig:
    controller_address: str
    user_address: str
    start_block: Union[int, str, None] = None
    end_block: Union[int, str, None] = None
    duration: int | None = None
    block_step: int = 5
    auto_fetch_events: bool = True
    events: list[dict] = field(default_factory=list)
    output_base: Path | None = None
    etherscan_api_key: str | None = None
    rpc_url: str | None = None
    max_workers: int = 8


# =============================================================================
# Helper functions
# =============================================================================

_BAR_CHARS = " ▏▎▍▌▋▊▉█"

def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Return a smooth text progress bar using partial-fill characters."""
    pct = current / total if total > 0 else 0
    full_width = pct * width
    filled = int(full_width)
    frac = full_width - filled
    partial = _BAR_CHARS[int(frac * 8)]
    empty = width - filled - 1
    if filled >= width:
        bar = "█" * width
    else:
        bar = "█" * filled + partial + " " * max(empty, 0)
    return f"▐{bar}▌ {pct * 100:5.1f}%"


def get_event_style(action: str) -> dict:
    """Return plotting style for an event action with sensible defaults."""
    if action in EVENT_STYLE:
        return EVENT_STYLE[action]
    if action and "borrow" in action:
        return {"color": "#1E90FF", "linestyle": ":", "linewidth": 1.0}
    if action and "repay" in action:
        return {"color": "#00FF00", "linestyle": ":", "linewidth": 1.0}
    return {"color": "#00FF00", "linestyle": ":", "linewidth": 1.0}


def get_health_color(health_val):
    """Return (fill_color, edge_color) tuple that transitions from red → green."""
    if health_val < 0:
        return "#CC0000", "#990000"
    elif health_val < 2:
        return "#FF4444", "#CC0000"
    elif health_val < 3.5:
        t = (health_val - 2) / 1.5
        r = int(255)
        g = int(68 + (165 - 68) * t)
        b = int(68)
        return f"#{r:02X}{g:02X}{b:02X}", f"#{int(r*0.8):02X}{int(g*0.8):02X}{int(b*0.8):02X}"
    elif health_val < 5:
        t = (health_val - 3.5) / 1.5
        r = int(255)
        g = int(165 + (255 - 165) * t)
        b = int(68 + (215 - 68) * t)
        return f"#{r:02X}{g:02X}{b:02X}", f"#{int(r*0.8):02X}{int(g*0.8):02X}{int(b*0.8):02X}"
    else:
        return "#4CAF50", "#2E7D32"


def _hex_to_rgba(hex_color, alpha=0.6):
    """Convert a hex color string to an RGBA tuple (0-1 floats)."""
    h = hex_color.lstrip('#')
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, alpha)


def _fill_band_column(img, col, bands, y_min, y_max, chart_colors):
    """Fill one column of the bands image array for a single frame.

    This replaces per-frame barh() calls with direct pixel writes into a
    numpy RGBA array, which is then displayed via a single imshow artist
    that is updated with set_data() each frame — O(1) instead of O(N).
    """
    h = img.shape[0]
    scale = h / (y_max - y_min)

    coll_fill = _hex_to_rgba(chart_colors['collateral']['fill'], 0.6)
    coll_edge = _hex_to_rgba(chart_colors['collateral']['edge'], 0.8)
    crvusd_fill = _hex_to_rgba(chart_colors['crvusd']['fill'], 0.6)
    crvusd_edge = _hex_to_rgba(chart_colors['crvusd']['edge'], 0.8)

    for band in bands:
        min_p = band['min_price']
        max_p = band['max_price']
        crvusd_amount = band['crvusd_amount']
        collateral_val = band['collateral_as_crvusd']
        total_val = band.get('band_total_value', band['total_collateral_value'])

        if total_val <= 0:
            continue

        # Map prices to pixel rows (row 0 = y_max = top of image)
        top_row = max(0, int((y_max - max_p) * scale))
        bot_row = min(h, int((y_max - min_p) * scale))

        if top_row >= bot_row:
            continue

        if collateral_val > 0 and crvusd_amount > 0:
            coll_pct = collateral_val / total_val
            # Collateral at bottom (higher row indices), crvUSD at top
            split_row = bot_row - int((bot_row - top_row) * coll_pct)
            img[top_row:split_row, col] = crvusd_fill
            img[split_row:bot_row, col] = coll_fill
            # Edge lines at boundaries
            if bot_row - top_row >= 3:
                img[top_row, col] = crvusd_edge
                img[split_row, col] = coll_edge
                img[bot_row - 1, col] = coll_edge
        elif collateral_val > 0:
            img[top_row:bot_row, col] = coll_fill
            if bot_row - top_row >= 3:
                img[top_row, col] = coll_edge
                img[bot_row - 1, col] = coll_edge
        elif crvusd_amount > 0:
            img[top_row:bot_row, col] = crvusd_fill
            if bot_row - top_row >= 3:
                img[top_row, col] = crvusd_edge
                img[bot_row - 1, col] = crvusd_edge


def _draw_band_strip(ax, left_pos, bar_width, band, chart_colors):
    """Draw one column of the band-composition heatmap on *ax*.

    Each band is split into a collateral portion (bottom, blue) and a crvUSD
    portion (top, green), proportional to their value.  If only one asset is
    present the full band height is used.
    """
    min_price = band['min_price']
    max_price = band['max_price']
    avg_price = band['avg_price']
    crvusd_amount = band['crvusd_amount']
    collateral_value = band['collateral_as_crvusd']
    total_value_for_bars = band.get('band_total_value', band['total_collateral_value'])

    if total_value_for_bars <= 0:
        return

    if collateral_value > 0 and crvusd_amount > 0:
        # Mixed band — split proportionally
        collateral_percent = collateral_value / total_value_for_bars
        collateral_end_price = min_price + (max_price - min_price) * collateral_percent

        ax.barh(
            (min_price + collateral_end_price) / 2,
            bar_width,
            left=left_pos,
            height=(collateral_end_price - min_price),
            color=chart_colors['collateral']['fill'],
            alpha=0.6,
            edgecolor=chart_colors['collateral']['edge'],
            linewidth=0.5,
        )
        ax.barh(
            (collateral_end_price + max_price) / 2,
            bar_width,
            left=left_pos,
            height=(max_price - collateral_end_price),
            color=chart_colors['crvusd']['fill'],
            alpha=0.6,
            edgecolor=chart_colors['crvusd']['edge'],
            linewidth=0.5,
        )
    elif collateral_value > 0:
        ax.barh(
            avg_price, bar_width, left=left_pos,
            height=(max_price - min_price),
            color=chart_colors['collateral']['fill'], alpha=0.6,
            edgecolor=chart_colors['collateral']['edge'], linewidth=0.5,
        )
    elif crvusd_amount > 0:
        ax.barh(
            avg_price, bar_width, left=left_pos,
            height=(max_price - min_price),
            color=chart_colors['crvusd']['fill'], alpha=0.6,
            edgecolor=chart_colors['crvusd']['edge'], linewidth=0.5,
        )


def _fmt_event_line(event, event_data):
    """Format a single event line for the bottom-left event info panel."""
    action = event.get('action', 'event')
    block_num = event.get('blocknumber', 0)

    if action == 'closed':
        label = "loan closed"
        icon = EVENT_ICONS.get(action, '•')
        amount_text = "-"
        health_text = "-"
    else:
        label = EVENT_LABELS.get(action, action.replace('_', ' ').title())
        icon = EVENT_ICONS.get(action, '•')
        amount = event.get('amount', 0.0)
        amount_text = f"{amount:>14,.2f}"
        health_inc = event_data.get('health_increase')
        health_text = f"{health_inc:+.2f}" if health_inc is not None else "  N/A"

    return f"{icon} {label:<14} {block_num:>10,} {amount_text:>14} {health_text:>8}"


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(config: PipelineConfig, log_fn=print) -> Path:
    """Run the full visualization pipeline and return the path to the output MP4."""

    events = list(config.events)
    output_base = config.output_base or OUTPUT_BASE

    # --- 0. Connect to RPC and resolve block range --------------------------
    rpc = RPCManager(log_fn=log_fn, rpc_url=config.rpc_url)

    if config.auto_fetch_events and not events:
        log_fn("Fetching loan events from Curve Prices API...")
        events = fetch_events(config.controller_address, config.user_address,
                              rpc.w3, rpc_manager=rpc, log_fn=log_fn)

    esk = config.etherscan_api_key
    log_fn("Resolving block range...")
    if config.start_block is not None and config.end_block is not None:
        start_block = resolve_block(config.start_block, rpc.w3, log_fn=log_fn, etherscan_api_key=esk)
        end_block = resolve_block(config.end_block, rpc.w3, log_fn=log_fn, etherscan_api_key=esk)
    elif config.start_block is not None and config.duration is not None:
        start_block = resolve_block(config.start_block, rpc.w3, log_fn=log_fn, etherscan_api_key=esk)
        end_block = start_block + config.duration
    elif config.end_block is not None and config.duration is not None:
        end_block = resolve_block(config.end_block, rpc.w3, log_fn=log_fn, etherscan_api_key=esk)
        start_block = end_block - config.duration
    elif config.start_block is None and config.end_block is None and config.duration is not None:
        end_block = rpc.w3.eth.block_number
        start_block = end_block - config.duration
    else:
        raise ValueError(
            "Invalid block range config. Set start_block + end_block, "
            "or start_block + duration, or end_block + duration."
        )

    # Filter events to the resolved block range
    events = [e for e in events if start_block <= e['blocknumber'] <= end_block]

    block_step = config.block_step
    num_blocks = (end_block - start_block) // block_step + 1
    log_fn(f"  Range: {start_block} -> {end_block} "
           f"({end_block - start_block} blocks, {num_blocks} frames, step {block_step})")

    # --- 1. Initialize loan getter (fetches AMM address, coins, etc.) -------
    loan = CurveLoanGetter(
        rpc.w3,
        block_number=start_block,
        controller_address=config.controller_address,
        user_address=config.user_address,
        rpc_manager=rpc,
        log_fn=log_fn,
    )
    base_symbol = loan.coins[0]['symbol']
    collateral_symbol = loan.coins[1]['symbol']
    market_label = f"{collateral_symbol}-{base_symbol}"

    # --- 2. Create output directory -----------------------------------------
    output_base.mkdir(exist_ok=True)
    user_short = config.user_address[:6]
    run_name = f"{date.today()}_{collateral_symbol}-{base_symbol}_{user_short}"
    run_dir = output_base / run_name
    if run_dir.exists():
        n = 2
        while (output_base / f"{run_name}_{n}").exists():
            n += 1
        run_dir = output_base / f"{run_name}_{n}"
    run_dir.mkdir()
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()

    file_str = f"{base_symbol}_{collateral_symbol}"
    mp4_filename = run_dir / "video.mp4"

    # --- 3. FIRST PASS: fetch on-chain data for every sampled block ---------
    log_fn("=" * 60)
    log_fn("FIRST PASS: Fetching all data from blockchain...")
    log_fn("=" * 60)

    # If there's a "closed" or "liquidated" event, stop fetching after it
    close_loan_block = None
    for event in events:
        if event.get('action') in ['closed', 'liquidated']:
            close_loan_block = event['blocknumber']
            break

    # Pre-compute list of blocks to fetch
    blocks_to_fetch = []
    for block in range(start_block, end_block + 1, block_step):
        if close_loan_block is not None and block > close_loan_block:
            break
        blocks_to_fetch.append(block)

    num_blocks_to_fetch = len(blocks_to_fetch)

    # Create separate Web3 instances per RPC URL so load is distributed
    # from the start (not via failover-switching, which is chaotic under
    # heavy parallel load).
    w3_pool = [Web3(Web3.HTTPProvider(url)) for url in rpc.urls]
    max_workers = min(config.max_workers, len(w3_pool) * 2)
    log_fn(f"  Fetching {num_blocks_to_fetch} blocks with {max_workers} workers "
           f"across {len(w3_pool)} RPCs...")

    start_time = time.time()
    all_block_data = []
    completed_count = 0
    log_interval = max(1, num_blocks_to_fetch // 50)

    def _fetch_one(block_and_w3):
        block, w3 = block_and_w3
        return loan.fetch_block_data(block, w3=w3)

    # Round-robin assign blocks to Web3 instances
    block_assignments = [
        (b, w3_pool[i % len(w3_pool)])
        for i, b in enumerate(blocks_to_fetch)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_block = {
            executor.submit(_fetch_one, pair): pair[0]
            for pair in block_assignments
        }

        for future in concurrent.futures.as_completed(future_to_block):
            block = future_to_block[future]
            completed_count += 1

            try:
                data = future.result()
                all_block_data.append(data)
            except BlockNotFound:
                log_fn(f"  Block {block} not found (future block?). Skipping.")
            except ContractLogicError as e:
                error_msg = str(e)
                if 'Multicall3: call failed' in error_msg or 'execution reverted' in error_msg:
                    log_fn(f"  Contract call failed at block {block} (loan closed?). Skipping.")
                else:
                    raise
            except Exception as e:
                if close_loan_block is not None and block >= close_loan_block:
                    log_fn(f"  Error at block {block} near close event. Skipping.")
                else:
                    raise

            if (completed_count == 1
                    or completed_count == num_blocks_to_fetch
                    or completed_count % log_interval == 0):
                elapsed = time.time() - start_time
                eta = (elapsed / completed_count) * (num_blocks_to_fetch - completed_count)
                log_fn(f"{_progress_bar(completed_count, num_blocks_to_fetch)} "
                       f"| Fetched {completed_count}/{num_blocks_to_fetch} blocks "
                       f"| ETA: {eta:.1f}s")

    # Sort by block number (futures complete out of order)
    all_block_data.sort(key=lambda d: d['block'])

    # Accumulate price/band/health ranges for axis scaling
    all_prices = []
    all_band_nums = set()
    all_health_values = []
    for bd in all_block_data:
        all_prices.append(bd['price_oracle'])
        all_health_values.append(bd['health'])
        for band in bd['bands']:
            all_prices.append(band['min_price'])
            all_prices.append(band['max_price'])
            all_band_nums.add(band['band_num'])

    fetch_elapsed = time.time() - start_time
    log_fn(f"\n  Data fetching complete. {len(all_block_data)} blocks in {fetch_elapsed:.1f}s")

    # Save cached block data so re-rendering doesn't require re-fetching
    data_filename = run_dir / "block_data.json"
    block_data_output = {
        "controller_address": config.controller_address,
        "llamma_address": loan.llamma_address,
        "user_address": config.user_address,
        "market": market_label,
        "base_symbol": base_symbol,
        "collateral_symbol": collateral_symbol,
        "start_block": start_block,
        "end_block": end_block,
        "block_step": block_step,
        "num_frames": len(all_block_data),
        "generated": str(date.today()),
        "events": events,
        "blocks": all_block_data,
    }
    with data_filename.open("w") as f:
        json.dump(block_data_output, f, indent=2, default=str)
    log_fn(f"  Saved block data to {data_filename}")

    # --- 4. Compute fixed axis ranges (consistent across all frames) --------
    if len(all_prices) > 0:
        price_min = min(all_prices)
        price_max = max(all_prices)
        price_range = price_max - price_min
        price_center = (price_min + price_max) / 2
        # 2x multiplier gives breathing room so bands look compact
        expanded_range = price_range * 2.0
        y_min = price_center - expanded_range / 2
        y_max = price_center + expanded_range / 2
    else:
        y_min, y_max = 2700, 3200
        price_min, price_max = y_min, y_max

    log_fn(f"  Y-axis: {y_min:.2f} – {y_max:.2f} (data: {price_min:.2f} – {price_max:.2f})")
    log_fn(f"  Bands: {len(all_band_nums)} unique {sorted(all_band_nums)}")

    if len(all_health_values) > 0:
        health_max = max(all_health_values)
        health_min = min(all_health_values)
        if health_max > 0:
            health_y_max = health_max * 1.1
        else:
            health_y_max = max(0, health_max * 0.9) + 1
        if health_min < 0:
            health_y_min = health_min * 1.1
        elif health_min > 0:
            health_y_min = health_min * 0.9
        else:
            health_y_min = -1
        health_y_max = int(health_y_max) + (1 if health_y_max > 0 else 0)
        health_y_min = int(health_y_min) - (1 if health_y_min < 0 else 0)
    else:
        health_y_max, health_y_min = 5, -1

    # --- 5. Map events to the nearest fetched block index -------------------
    events_by_block_index = {}  # {frame_index: [{event, health_increase, ...}]}
    close_loan_block_index = None

    if events:
        fetched_blocks = [bd['block'] for bd in all_block_data]
        health_values = [bd['health'] for bd in all_block_data]

        for event in events:
            event_block = event['blocknumber']
            is_close_event = event.get('action') in ['closed', 'liquidated']

            closest_idx = min(range(len(fetched_blocks)),
                              key=lambda i: abs(fetched_blocks[i] - event_block))

            if is_close_event:
                close_loan_block_index = closest_idx
                event_type = "Liquidation" if event.get('action') == 'liquidated' else "Close loan"
                log_fn(f"  {event_type} event at block {event_block} "
                       f"-> frame {closest_idx} (block {fetched_blocks[closest_idx]})")

            health_after = health_values[closest_idx] if closest_idx < len(health_values) else None
            health_before = health_values[closest_idx - 1] if closest_idx > 0 else None
            health_increase = (health_after - health_before
                               if health_after is not None and health_before is not None
                               else None)

            events_by_block_index.setdefault(closest_idx, []).append({
                'event': event,
                'health_increase': health_increase,
                'health_before': health_before,
                'health_after': health_after,
            })

    # --- 6. SECOND PASS: render frames (incremental — figure persists) ------
    #
    # Performance: instead of recreating the figure each frame, we create it
    # once and update only what changes:
    #   - Band strips: draw ONE new column per frame (they accumulate on ax_main)
    #   - Price line: update via set_data()
    #   - Health bar / stats / events text: update artist properties in-place
    #   - Title: set_title() each frame
    # This avoids O(N^2) barh redraws and cuts rendering time ~3x.

    log_fn("\n" + "=" * 60)
    log_fn("SECOND PASS: Generating images from stored data...")
    log_fn("=" * 60)

    price_history = []
    time_indices = []
    image_start_time = time.time()

    # Create figure + axes once
    dpi_num = 150
    fig = plt.figure(figsize=(10.5, 5.6), dpi=dpi_num)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.3], width_ratios=[6, 1.4],
                          hspace=0.25, wspace=0.25)
    ax_main = fig.add_subplot(gs[0, 0])     # main chart (bands + price)
    ax_health = fig.add_subplot(gs[0, 1])   # health bar (right)
    ax_events = fig.add_subplot(gs[1, 0])   # event log (bottom-left)
    ax_stats = fig.add_subplot(gs[1, 1])    # stats box (bottom-right)
    ax_events.axis('off')
    ax_stats.axis('off')

    # Static setup (done once, never cleared)
    ax_main.grid(axis='y', linestyle='--', alpha=0.5)
    ax_main.grid(axis='x', linestyle='--', alpha=0.3)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlim(0, len(all_block_data))
    ax_main.set_xlabel("Time (Block Steps)")
    ax_main.set_ylabel(f"{collateral_symbol} Price ({base_symbol})")
    ax_main.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.2f}'))

    # Price line — updated each frame via set_data()
    price_line, = ax_main.plot([], [], color="#3162F4", linestyle='-', linewidth=2.5,
                               marker='o', markersize=2, alpha=0.8)

    # Legend — built once upfront (includes all event types from events)
    legend_handles = [
        Line2D([], [], color="#3162F4", linewidth=2.5, marker='o', markersize=2, alpha=0.8),
        Patch(facecolor=CHART_COLORS['collateral']['fill'], edgecolor=CHART_COLORS['collateral']['edge'], alpha=0.6),
        Patch(facecolor=CHART_COLORS['crvusd']['fill'], edgecolor=CHART_COLORS['crvusd']['edge'], alpha=0.6),
    ]
    legend_labels = ["Oracle Price", f"Collateral {collateral_symbol}", f"Collateral {base_symbol}"]
    if events:
        seen_actions = set()
        for event in events:
            action = event.get('action')
            if action in {'closed', 'liquidated'} or action in seen_actions:
                continue
            seen_actions.add(action)
            label = EVENT_LABELS.get(action, action.replace('_', ' ').title())
            style = get_event_style(action)
            legend_handles.append(Line2D([], [], color=style["color"],
                                        linestyle=style["linestyle"],
                                        linewidth=style["linewidth"], alpha=0.8))
            legend_labels.append(label)
    ax_main.legend(legend_handles, legend_labels, loc='upper left')

    # Health axis
    ax_health.set_xlim(-0.8, 0.8)
    ax_health.set_xticks([])
    ax_health.set_ylim(health_y_min, health_y_max)
    ax_health.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    ax_health.set_ylabel("Health")

    # Reusable artists — updated in-place each frame (no clearing needed)
    health_bar_artist = ax_health.bar(0, 0, width=0.6, color="#4CAF50",
                                      edgecolor="#2E7D32", linewidth=0.8)[0]
    health_text_artist = ax_health.text(0, 0, '', ha='center', va='bottom',
                                        fontsize=9, fontweight='bold', color="#2E7D32")
    stats_text_artist = ax_stats.text(0, 1, '', transform=ax_stats.transAxes,
                                      fontsize=8, verticalalignment='top',
                                      horizontalalignment='left',
                                      family='monospace', color="#1d1d1f")
    events_text_artist = ax_events.text(0, 1, '', transform=ax_events.transAxes,
                                        fontsize=7.3, verticalalignment='top',
                                        horizontalalignment='left',
                                        family='monospace', color="#1d1d1f")

    fig.tight_layout(rect=[0.03, 0.02, 0.98, 0.97])

    # Bands image: numpy RGBA array rendered via imshow instead of accumulating
    # barh artists.  Each frame fills one column — O(1) per frame, not O(N).
    bands_pixel_h = 600
    bands_image = np.zeros((bands_pixel_h, len(all_block_data), 4), dtype=np.float32)
    bands_im = ax_main.imshow(
        bands_image, extent=[0, len(all_block_data), y_min, y_max],
        aspect='auto', origin='upper', interpolation='nearest', zorder=0,
    )

    # Track which events have been drawn (they persist on ax_main once added)
    plotted_event_indices = set()
    max_band_idx = close_loan_block_index if close_loan_block_index is not None else len(all_block_data)

    for idx, block_data in enumerate(all_block_data):
        cycle = idx + 1
        block = block_data['block']

        total_images = len(all_block_data)
        render_interval = max(1, total_images // 50)
        if cycle == 1 or cycle == total_images or cycle % render_interval == 0:
            if cycle == 1:
                log_fn(f"{_progress_bar(0, total_images)} | Rendering frame 1/{total_images} (block {block})")
            else:
                elapsed_time = time.time() - image_start_time
                eta = (elapsed_time / (cycle - 1)) * (total_images - cycle + 1)
                log_fn(f"{_progress_bar(cycle, total_images)} | Rendering frame {cycle}/{total_images} (block {block}) | ETA: {eta:.1f}s")

        if close_loan_block_index is not None and idx == close_loan_block_index:
            event_type = "liquidation" if any(e.get('action') == 'liquidated' for e in events if e.get('blocknumber') == block) else "close loan"
            log_fn(f"\n  Reached {event_type} event at block {block}. Last image.")

        price_oracle = block_data['price_oracle']
        health = block_data['health']
        block_date = block_data['block_date']
        bands = block_data['bands']

        loan.get_base_price = block_data['get_base_price']
        loan.A = block_data['A']

        price_history.append(price_oracle)
        time_indices.append(idx)

        # Band strips: fill one column of the numpy image array (O(1) per frame)
        if idx < max_band_idx:
            _fill_band_column(bands_image, idx, bands, y_min, y_max, CHART_COLORS)
            bands_im.set_data(bands_image)

        # Price line: update full history
        price_line.set_data(time_indices, price_history)

        # Title: block number + UTC timestamp
        ax_main.set_title(
            f"{market_label} Mint Market - Time Series, Block: {block}, Time: {block_date}",
            pad=12)

        # Event markers: add once when they first appear, then they persist
        if events:
            for event_idx, event_list in events_by_block_index.items():
                if event_idx > idx or event_idx in plotted_event_indices:
                    continue
                plotted_event_indices.add(event_idx)
                for event_data in event_list:
                    event = event_data['event']
                    action = event.get('action')
                    if action in ['closed', 'liquidated']:
                        closed_price = price_history[event_idx] if event_idx < len(price_history) else price_history[-1]
                        is_liquidation = action == 'liquidated'
                        marker_color = '#FF0000'
                        marker_style = 'X' if is_liquidation else 'o'
                        marker_size = 8 if is_liquidation else 7
                        event_text = 'liquidated' if is_liquidation else 'loan closed'

                        ax_main.plot(
                            event_idx, closed_price, marker_style,
                            color=marker_color, markersize=marker_size,
                            markeredgecolor=marker_color,
                            markeredgewidth=1.5 if is_liquidation else 1,
                            zorder=10, label="_nolegend_")
                        ax_main.text(
                            event_idx, closed_price + (y_max - y_min) * 0.05,
                            event_text, fontsize=8, color=marker_color,
                            fontweight='bold', ha='center', va='bottom', zorder=6,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor=marker_color, linewidth=1.0, alpha=0.9))
                    else:
                        style = get_event_style(action)
                        ax_main.axvline(
                            x=event_idx, color=style["color"],
                            linestyle=style["linestyle"],
                            linewidth=style["linewidth"],
                            alpha=0.6, zorder=2)

            # Event log panel (bottom-left) — shows last 5 events
            event_lines = []
            for eidx, elist in events_by_block_index.items():
                if eidx > idx:
                    continue
                for ed in elist:
                    event_lines.append(_fmt_event_line(ed['event'], ed))
            if len(event_lines) > 5:
                event_lines = event_lines[-5:]
            if event_lines:
                header = f"{'':1} {'Event':<14} {'Block':>10} {'Amount':>14} {'Health':>8}"
                separator = "-" * len(header)
                events_text_artist.set_text("\n".join([header, separator] + event_lines))
            else:
                events_text_artist.set_text('')
        else:
            events_text_artist.set_text('')

        # Health bar (right panel)
        health_fill_color, health_edge_color = get_health_color(health)
        health_bar_artist.set_height(health)
        health_bar_artist.set_color(health_fill_color)
        health_bar_artist.set_edgecolor(health_edge_color)

        if health >= 0:
            text_y = health + (health_y_max - health_y_min) * 0.02
            va = 'bottom'
        else:
            text_y = health - (health_y_max - health_y_min) * 0.02
            va = 'top'
        health_text_artist.set_text(f'{health:.2f}')
        health_text_artist.set_position((0, text_y))
        health_text_artist.set_va(va)
        health_text_artist.set_color(health_edge_color)

        # Stats box (bottom-right)
        current_debt = block_data.get('debt', 0) or 0
        current_collateral = block_data.get('total_collateral_value') or 0
        current_native = block_data.get('collateral_native', 0)
        current_crvusd = block_data.get('collateral_crvusd', 0)
        current_bands_state = block_data.get('num_bands_state', len(block_data.get('bands', [])))

        ltv = (current_debt / current_collateral) if current_collateral > 0 else None
        ltv_text = f"{ltv*100:,.2f}%" if ltv is not None else "N/A"

        stats_text_artist.set_text("\n".join([
            f"Bands:       {current_bands_state}",
            f"LTV:         {ltv_text}",
            f"Debt:        {current_debt:,.2f}",
            f"Collateral:  ${current_collateral:,.2f}",
            f"  - {collateral_symbol:<6} {current_native:,.4f}",
            f"  - crvUSD:  {current_crvusd:,.4f}",
        ]))

        fig.savefig(frames_dir / f"{file_str}_{cycle}.png", dpi=dpi_num)

        if close_loan_block_index is not None and idx == close_loan_block_index:
            event_type = "liquidation" if any(e.get('action') == 'liquidated' for e in events if e.get('blocknumber') == block) else "close loan"
            log_fn(f"  Stopping after {event_type} event (block {block})")
            break

    plt.close(fig)

    # --- 7. Encode video with ffmpeg ----------------------------------------
    image_elapsed = time.time() - image_start_time
    total_elapsed = time.time() - start_time
    log_fn(f"\n  Image generation: {image_elapsed:.1f}s")
    log_fn(f"  Data fetching:    {fetch_elapsed:.1f}s")
    log_fn(f"  Total:            {total_elapsed:.1f}s")

    cmd = [
        "ffmpeg",
        "-start_number", "1",
        "-framerate", "50",
        "-i", str(frames_dir / f"{file_str}_%d.png"),
        "-vf", "scale=1280:-2:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-movflags", "+faststart",
        str(mp4_filename),
    ]
    result = subprocess.run(cmd)

    # Clean up frames after successful encoding
    if result.returncode == 0 and mp4_filename.exists():
        shutil.rmtree(frames_dir)
        log_fn(f"  Cleaned up frames")
    else:
        log_fn(f"  ffmpeg failed (exit {result.returncode}), keeping frames in {frames_dir}")

    log_fn(f"  Output: {run_dir}")
    return mp4_filename


if __name__ == "__main__":
    config = PipelineConfig(
        controller_address=CONTROLLER_ADDRESS,
        user_address=USER_ADDRESS,
        start_block=START_BLOCK,
        end_block=END_BLOCK,
        duration=DURATION,
        block_step=BLOCK_STEP,
        auto_fetch_events=AUTO_FETCH_EVENTS,
        events=EVENTS,
        output_base=OUTPUT_BASE,
        max_workers=MAX_WORKERS,
    )
    run_pipeline(config)
